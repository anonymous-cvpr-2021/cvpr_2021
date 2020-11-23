import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import pytorch_ssim

import argparse
from time import time

from generator import *
from segmentation_network import *
from discriminator import *
from warper import *

from data_generator import *

from metric import *
from diceloss import *
from gradient_loss import *
from focalloss import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_train(models, save_file_name):
	#--- copy models to cpu  and get state-dict---#
	generator = models['gen']
	segmentator = models['seg']
	discriminator = models['dis']
	models_state_dict = {'gen': generator.state_dict(), 'seg': segmentator.state_dict(), 'dis': discriminator.state_dict()}

	torch.save(models_state_dict, save_file_name)	
	return

def load_train(file_name):
	return torch.load(file_name)

def train_epoch(models, optimizer, loader, batch_size, device, scale_factor=0.25):
	lambda_1 = 0.5
	lambda_2 = 1
	lambda_3 = 10
	lambda_4 = 1
	
	warper = Warper(device)
	segmentator = models['seg'].train()
	generator = models['gen'].train()
	discriminator = models['dis'].train()
	
	dis_criterion = torch.nn.CrossEntropyLoss()
	ssim_criterion = pytorch_ssim.SSIM3D(window_size = 11)
	classes_weights = [1/33, 1/33, 1/33, 10/33, 10/33, 10/33]
	dice_criterion = Dice_Loss()
	smooth_criterion = Gradient_Loss()
	focal_criterion = Focal_Loss(0.7, 0.3, 4/3, weights=classes_weights)
	
	optimizer_gs = optimizer['gs']
	optimizer_d  = optimizer['d']
	
	run_dis_loss = 0
	run_adv_loss = 0
	run_inv_loss = 0
	run_smooth = 0
	run_div_loss = 0
	run_segmentation_loss = 0
	run_dice_gt = np.zeros(6)
	run_dice_pd = np.zeros(6)
	
	start_time = time()
	
	for step, (x, y, x_1, x_2, y_1, y_2, s, k_1, k_2, inter_subject, intra_subject) in enumerate(loader):
		#print(step)
		#Load training example to device
		x, y, k_1, k_2 = x.to(device), y.to(device), k_1.to(device), k_2.to(device)
		y_1, y_2 = y_1.to(device), y_2.to(device)
		s, inter_subject, intra_subject = s.to(device), inter_subject.to(device), intra_subject.to(device)
		
		#train discriminator
		optimizer_d.zero_grad()
		
		pred = discriminator(y_1, y_2)
		dis_loss = dis_criterion(pred, s)

		dis_loss.backward()
		optimizer_d.step()

		run_dis_loss += dis_loss.item()

		#train generator and segmentation
		optimizer_gs.zero_grad()
		#generate deformed image
		_f_1_f, _f_1_b = generator(x, k_1)
		f_1_f, f_1_b =  scale_factor * _f_1_f, scale_factor * _f_1_b
		z_1 = warper(x, f_1_f)
		y_1 = warper(y, f_1_f)
		
		_f_2_f, _f_2_b = generator(x, k_2)
		f_2_f, f_2_b =  scale_factor * _f_2_f, scale_factor * _f_2_b
		z_2 = warper(x, f_2_f)
		y_2 = warper(y, f_2_f)	

		# Reconstruct image
		x_r_1 = warper(z_1, f_1_b)
		x_r_2 = warper(z_2, f_2_b)
		
		y_r_1 = warper(y_1, f_1_b)
		y_r_2 = warper(y_2, f_2_b)	
		
		# Segmentation
		hat_y_d_1 = segmentator(z_1)
		hat_y_d_2 = segmentator(z_2)
		hat_y_1 = warper(hat_y_d_1, f_1_b)
		hat_y_2 = warper(hat_y_d_2, f_2_b)
		
		# Discriminator
		pred = discriminator(hat_y_1, y)
		
		# compute loss
		# Invetibility Loss
		res_1 = -ssim_criterion(x_r_1, x)
		res_2 = -ssim_criterion(x_r_2, x)
		reconstruction_ssim_loss = 1/2 * (res_1 + res_2)		
		dice_1 = dice_criterion(y_r_1, y)
		dice_2 = dice_criterion(y_r_2, y)
		reconstruction_dice_loss = 1/2 * (dice_1 + dice_2)		
		inv_loss = 1/2 * (reconstruction_ssim_loss + reconstruction_dice_loss)
		
		# Diversity Loss
		distortion_loss = ssim_criterion(z_1, z_2)
		distortion_dice_loss = -dice_criterion(y_1, y_2)		
		div_loss = 1/2 * (distortion_loss + distortion_dice_loss)
		
		# Smooth
		smooth_1 = smooth_criterion(_f_1_f)
		smooth_2 = smooth_criterion(_f_2_f)
		smooth_3 = smooth_criterion(_f_1_b)
		smooth_4 = smooth_criterion(_f_2_b)
		smooth = 0.25 * (smooth_1 + smooth_2 + smooth_3 + smooth_4)
		
		# Segmentation Loss
		seg_loss_1 = focal_criterion(hat_y_1, y)
		seg_loss_2 = focal_criterion(hat_y_2, y)
		segmentation_loss = 1/2 * (seg_loss_1 + seg_loss_2)

		# Adversarial Loss
		adv_loss = dis_criterion(pred, inter_subject)
		
		total_loss =  segmentation_loss + lambda_1 * adv_loss\
					+ lambda_2 * inv_loss + lambda_3 * smooth + lambda_4 * div_loss
					

		total_loss.backward()
		optimizer_gs.step()
		
		# compute reconstructed ground truth dice score
		dice_score_gt_1 = compute_dice_score(y_r_1, y, device)
		dice_score_gt_2 = compute_dice_score(y_r_2, y, device)
		dice_score_gt = 0.5 * (dice_score_gt_1 + dice_score_gt_2)
		run_dice_gt += dice_score_gt
		# compute reconstructed predicted segmap dice score
		dice_score_pd_1 = compute_dice_score(hat_y_1, y, device)
		dice_score_pd_2 = compute_dice_score(hat_y_2, y, device)
		dice_score_pd = 0.5 * (dice_score_pd_1 + dice_score_pd_2)
		run_dice_pd += dice_score_pd
		
		# Accumulate loss values
		run_adv_loss += adv_loss.item()
		run_inv_loss += inv_loss.item()
		run_smooth_loss += smooth.item()
		run_div_loss += div_loss.item()
		run_segmentation_loss += segmentation_loss.item()

	dur = time() - start_time
				
	#Logging values
	models = {'gen': generator, 'seg': segmentator, 'dis': discriminator}
	optimizer = {'gs': optimizer_gs, 'd': optimizer_d}
	
	avg_dis_loss = run_dis_loss / (step + 1)
	avg_inv_loss = run_inv_loss / (step + 1)
	avg_div_loss = run_div_loss / (step + 1)
	avg_smooth_loss = run_smooth / (step + 1)
	avg_adv_loss = run_adv_loss / (step + 1)
	avg_segmentation_loss = run_segmentation_loss / (step + 1)
	avg_dice_gt = run_dice_gt / (step + 1)
	avg_dice_pd = run_dice_pd / (step + 1)
	
	#Print log
	print('invertibility_loss:      {:.4f} -- diversity_loss:     {:.4f}'\
			.format(avg_inv_loss, avg_div_loss))
	print('reconstruction_dice_score:{:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'\
		.format(avg_dice_gt[0],avg_dice_gt[1],avg_dice_gt[2],avg_dice_gt[3],avg_dice_gt[4],avg_dice_gt[5]))
	print('segmentation_loss:        {:.4f}'\
			.format(avg_segmentation_loss))
	print('Segmentation_dice_score:  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'\
		.format(avg_dice_pd[0],avg_dice_pd[1],avg_dice_pd[2],avg_dice_pd[3],avg_dice_pd[4],avg_dice_pd[5]))
	print('smooth:                   {:.4f}'.format(avg_smooth))
	print('discriminator:                   {:.4f}'.format(avg_dis_loss))
	print('adversarial  :                   {:.4f}'.format(avg_adv_loss))
	print('duration:{:.0f}'.format(dur))

	return models, optimizer

def val_epoch(models, loader, batch_size, device, scale_factor=0.25):
	warper = Warper(device)
	segmentator = models['seg'].train()
	generator = models['gen'].train()
	discriminator = models['dis'].train()
	
	dis_criterion = torch.nn.CrossEntropyLoss()
	ssim_criterion = pytorch_ssim.SSIM3D(window_size = 11)
	classes_weights = [1/33, 1/33, 1/33, 10/33, 10/33, 10/33]
	dice_criterion = Dice_Loss()
	smooth_criterion = Gradient_Loss()
	focal_criterion = Focal_Loss(0.7, 0.3, 4/3, weights=classes_weights)
	
	run_dis_loss = 0
	run_adv_loss = 0
	run_inv_loss = 0
	run_smooth = 0
	run_div_loss = 0
	run_segmentation_loss = 0
	run_dice_gt = np.zeros(6)
	run_dice_pd = np.zeros(6)

	start_time = time()
	with torch.no_grad():
		for step, (x, y, x_1, x_2, y_1, y_2, s, k_1, k_2, inter_subject, intra_subject) in enumerate(loader):
			#print(step)
			#Load training example to device
			x, y, k_1, k_2 = x.to(device), y.to(device), k_1.to(device), k_2.to(device)
			y_1, y_2 = y_1.to(device), y_2.to(device)
			s, inter_subject, intra_subject = s.to(device), inter_subject.to(device), intra_subject.to(device)		
			
			#Validate
			pred = discriminator(y_1, y_2)
			dis_loss = dis_criterion(pred, s)
			run_dis_loss += dis_loss.item()
			
			# Generate fake images
			_f_1_f, _f_1_b = generator(x, k_1)
			f_1_f, f_1_b =  scale_factor * _f_1_f, scale_factor * _f_1_b
			z_1 = warper(x, f_1_f)
			y_1 = warper(y, f_1_f)
			
			_f_2_f, _f_2_b = generator(x, k_2)
			f_2_f, f_2_b =  scale_factor * _f_2_f, scale_factor * _f_2_b
			z_2 = warper(x, f_2_f)
			y_2 = warper(y, f_2_f)
			
			# Reconstruct image
			x_r_1 = warper(z_1, f_1_b)
			x_r_2 = warper(z_2, f_2_b)

			y_r_1 = warper(y_1, f_1_b)
			y_r_2 = warper(y_2, f_2_b)

			# Segmentation
			hat_y_d_1 = segmentator(z_1)
			hat_y_d_2 = segmentator(z_2)
			hat_y_1 = warper(hat_y_d_1, f_1_b)
			hat_y_2 = warper(hat_y_d_2, f_2_b)
			
			pred = discriminator(hat_y_1, y)

			# Compute loss
			res_1 = -ssim_criterion(x_r_1, x)
			res_2 = -ssim_criterion(x_r_2, x)
			reconstruction_ssim_loss = 1/2 * (res_1 + res_2)		
			dice_1 = dice_criterion(y_r_1, y)
			dice_2 = dice_criterion(y_r_2, y)
			reconstruction_dice_loss = 1/2 * (dice_1 + dice_2)		
			inv_loss = 1/2 * (reconstruction_ssim_loss + reconstruction_dice_loss)
			
			distortion_loss = ssim_criterion(z_1, z_2)
			distortion_dice_loss = -dice_criterion(y_1, y_2)		
			div_loss = 1/2 * (distortion_loss + distortion_dice_loss)

			# Smooth
			smooth_1 = smooth_criterion(_f_1_f)
			smooth_2 = smooth_criterion(_f_2_f)
			smooth_3 = smooth_criterion(_f_1_b)
			smooth_4 = smooth_criterion(_f_2_b)
			smooth = 0.25 * (smooth_1 + smooth_2 + smooth_3 + smooth_4)

			# segmentation Loss
			seg_loss_1 = focal_criterion(hat_y_1, y)
			seg_loss_2 = focal_criterion(hat_y_2, y)
			segmentation_loss = 0.5 * (seg_loss_1 + seg_loss_2)
			
			adv_loss = dis_criterion(pred, inter_subject)
			
			# compute reconstructed ground truth dice score
			dice_score_gt_1 = compute_dice_score(y_r_1, y, device)
			dice_score_gt_2 = compute_dice_score(y_r_2, y, device)
			dice_score_gt = 0.5 * (dice_score_gt_1 + dice_score_gt_2)
			run_dice_gt += dice_score_gt
			# compute reconstructed predicted segmap dice score
			dice_score_pd_1 = compute_dice_score(hat_y_1, y, device)
			dice_score_pd_2 = compute_dice_score(hat_y_2, y, device)
			dice_score_pd = 0.5 * (dice_score_pd_1 + dice_score_pd_2)
			run_dice_pd += dice_score_pd

			run_adv_loss += adv_loss.item()
			run_inv_loss += inv_loss.item()
			run_smooth_loss += smooth.item()
			run_div_loss += div_loss.item()
			run_segmentation_loss += segmentation_loss.item()
			
	dur = time() - start_time
	
	avg_dis_loss = run_dis_loss / (step + 1)
	avg_inv_loss = run_inv_loss / (step + 1)
	avg_div_loss = run_div_loss / (step + 1)
	avg_smooth_loss = run_smooth / (step + 1)
	avg_adv_loss = run_adv_loss / (step + 1)
	avg_segmentation_loss = run_segmentation_loss / (step + 1)
	avg_dice_gt = run_dice_gt / (step + 1)
	avg_dice_pd = run_dice_pd / (step + 1)
	
	#Print log
	print('invertibility_loss:      {:.4f} -- diversity_loss:     {:.4f}'\
			.format(avg_inv_loss, avg_div_loss))
	print('reconstruction_dice_score:{:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'\
		.format(avg_dice_gt[0],avg_dice_gt[1],avg_dice_gt[2],avg_dice_gt[3],avg_dice_gt[4],avg_dice_gt[5]))
	print('segmentation_loss:        {:.4f}'\
			.format(avg_segmentation_loss))
	print('Segmentation_dice_score:  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'\
		.format(avg_dice_pd[0],avg_dice_pd[1],avg_dice_pd[2],avg_dice_pd[3],avg_dice_pd[4],avg_dice_pd[5]))
	print('smooth:                   {:.4f}'.format(avg_smooth))
	print('discriminator:                   {:.4f}'.format(avg_dis_loss))
	print('adversarial  :                   {:.4f}'.format(avg_adv_loss))
	print('duration:{:.0f}'.format(dur))

	return

def train(save_file, train_set, val_set, batch_size, num_epochs, device, save_dir, vis_im, learning_rate=1e-4):

	if os.path.exists(save_dir) == False:
			os.mkdir(save_dir)
	
	#Init data loader
	train_loader = data.DataLoader(train_set, batch_size = batch_size, num_workers = 32, pin_memory=True, shuffle = True)
	val_loader = data.DataLoader(val_set, batch_size = batch_size, num_workers = 32, pin_memory=True, shuffle = True)
	
	#Init models and optimizers
	segmentator = Segmentator().to(device)
	generator = Generator().to(device)
	#generator.apply(init_weights)
	discriminator = Discriminator(in_dims=6)

	_params_gs = [{"params": segmentator.parameters()}, {"params": generator.parameters()}]
	optimizer_gs = torch.optim.Adam(_params, lr = learning_rate)
	_params_d = [{"params": discriminator.parameters()}]
	optimizer_d = torch.optim.Adam(_params, lr = learning_rate)
	
	# Load save file
	if save_file == None:
		print('========================================================')
		print('=               == Train from ....... ==               =')
		print('========================================================')
		pretrained_segmentator = ''
		if os.path.exists(pretrained_segmentator) == True:
			segmentator.load_state_dict(torch.load(pretrained_segmentator))
			print('Pretrained segmentation network: {}'.format(pretrained_segmentator))
		
		pretrained_generator = ''
		if os.path.exists(pretrained_generator) == True:
			models_state_dict = torch.load(pretrained_generator)
			generator.load_state_dict(models_state_dict['gen'])
			print('Pretrained generator: {}'.format(pretrained_generator))
			
		pretrained_discriminator = ''
		if os.path.exists(pretrained_discriminator) == True:
			models_state_dict = torch.load(pretrained_discriminator)
			discriminator.load_state_dict(models_state_dict['dis'])
			print('Pretrained discriminator: {}'.format(pretrained_discriminator))
	else:
		print('========================================================')
		print('=  ==  Resume training from {}  ==  ='.format(save_file))
		print('========================================================')
		models_state_dict = load_train(save_file)
		
		segmentator.load_state_dict(models_state_dict['seg'])
		generator.load_state_dict(models_state_dict['gen'])
		discriminator.load_state_dict(models_state_dict['dis'])
	models = {'gen': generator, 'seg': segmentator, 'dis': discriminator}

	for epoch in range(0, num_epochs):

		print('========================================================')
		print('Train epoch:{}'.format(epoch+1))	
		# train
		
		models, optimizer = train_epoch(models, optimizer, train_loader, batch_size, device)
		
		# val
		print('Validate epoch:{}'.format(epoch+1)) 	
		val_epoch(models, val_loader, batch_size, device)
	 	
		# save train
		save_name = os.path.join(save_dir, 'checkpoint.pth')
		save_train(models, save_name)
		
	return models

def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--num_epochs', type=int, default=2000)
	parser.add_argument('--save_file', type=str, default=None)
	parser.add_argument('--save_dir', type='saved_model')
	args = parser.parse_args()
	
	save_file = args.save_file#None
	
	num_epochs = args.num_epochs #2000
	save_dir = args.save_dir #exp0'
	batch_size = args.batch_size #2	
	
	device = torch.device('cuda:0')
	
	train_set = ppmi_pairs_2(data_folder = './trainset')
	val_set = ppmi_pairs_2(data_folder = './valset')
	
	train(save_file, train_set, val_set, batch_size, num_epochs, device, save_dir)

if __name__ == '__main__':
	main()

