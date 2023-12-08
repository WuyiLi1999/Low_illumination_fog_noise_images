import numpy as np
import cv2 as cv
from scipy.sparse import csc_matrix
from tqdm import tqdm
import sys
from scipy.sparse import identity
from scipy.sparse.linalg import inv


class SoftMatting:
	def __init__(self, im, t, epsilon, lamb, window=1):
		self.im = im
		self.t = t
		self.window = window
		self.epsilon = epsilon
		self.lamb = lamb
		self.W, self.H, _ = im.shape
		self.N = self.W * self.H
		self.part2_matrix = np.zeros([self.W, self.H, self.window, self.window])
		self.miu_matrix = np.zeros([self.W, self.H, self.window])

	def __get_laplacian(self):
		row = np.array([])
		col = np.array([])
		data = np.array([])
		window_size = self.window * 2 + 1
		w_k = window_size ** 2
		U = self.epsilon / w_k * np.eye(3)
		indexs = np.arange(self.N).reshape(self.W, self.H)
		with tqdm(total=self.W - 2 * self.window, desc="Calculating L", unit="it", file=sys.stdout) as pbar:
			for i in range(self.W - 2 * self.window):
				for j in range(self.H - 2 * self.window):
					im_window = self.im[i:i + window_size, j:j + window_size, :]
					window = np.array([im_window[:, :, 0].reshape(w_k),
								  im_window[:, :, 1].reshape(w_k),
								  im_window[:, :, 2].reshape(w_k)])
					miu_k = np.mean(window, axis=1)
					diff = window - np.tile(miu_k, 9).reshape(9, 3).T
					cov = (np.dot(diff, diff.T) / w_k) + U
					L_elem = np.eye(w_k) - (1 + np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff)) / w_k
					x = indexs[i:i + window_size, j:j + window_size].flatten()
					x = np.tile(x, 9).reshape(9, 9)
					y = x.T
					row = np.append(row, x.flatten())
					col = np.append(col, y.flatten())
					data = np.append(data, L_elem.flatten())
				pbar.update()
		L = csc_matrix((data, (row, col)), shape=(self.N, self.N))
		return L

	def get_t(self):
		L = self.__get_laplacian().todense()
		# L = csc_matrix((self.N, self.N))
		t_reshaped = self.t.reshape(1, self.N)
		T = self.lamb * np.dot(t_reshaped, np.linalg.inv(L + self.lamb * np.eye(self.N)))
		T = T.reshape(self.W, self.H)
		return T

