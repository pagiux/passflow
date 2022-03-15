import math
import os
import pickle
import sys

import numpy as np
import torch
from torch.nn import functional as F

from tqdm import tqdm

from models import RealNVP, AffineTransform, MaskType


class PassFlow:
    def __init__(self, dim, dataset, architecture, lr, weight_decay, num_coupling=18, checkpoint_frequency=1,
                 mask_pattern=None, n_hidden=2, hidden_size=256, noise=0.1):
        self.dim = dim
        self.num_coupling = num_coupling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mask = ['left', 'right']
        if mask_pattern is None:
            mask_pattern = [MaskType.CHECKERBOARD] * self.num_coupling

        self.mask_pattern = ''.join(str(int(m)) for m in mask_pattern)

        flows = [AffineTransform(self.dim, self.device, mask[i % 2], mask_pattern[i], architecture,
                                 n_hidden=n_hidden, hidden_size=hidden_size) for i in range(self.num_coupling)]

        self.model = RealNVP(self.dim,
                             self.device,
                             flows)

        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=weight_decay)

        self.dataset = dataset
        self.checkpoint_frequency = checkpoint_frequency
        self.noise = noise

        self.sample_dir = 'samples'
        if not os.path.exists(os.path.join(os.getcwd(), self.sample_dir)):
            os.makedirs(os.path.join(os.getcwd(), self.sample_dir))

        self.checkpoint_dir = 'checkpoint'
        if not os.path.exists(os.path.join(os.getcwd(), self.checkpoint_dir)):
            os.makedirs(os.path.join(os.getcwd(), self.checkpoint_dir))

        self.current_epoch = 0

    def save(self, fname):
        torch.save({'epoch': self.current_epoch,
                    'optimizer': self.optimizer.state_dict(),
                    'net': self.model.state_dict()}, os.path.join(self.checkpoint_dir, fname))

    def load(self, fname):
        print('Loading model checkpoint {}'.format(fname))
        state_dicts = torch.load(os.path.join(self.checkpoint_dir, fname), map_location=self.device)
        self.current_epoch = state_dicts['epoch']
        self.model.load_state_dict(state_dicts['net'])
        try:
            self.optimizer.load_state_dict(state_dicts['optimizer'])
        except ValueError:
            print('Cannot load optimizer for some reason or other')

    def preprocess(self, x, reverse=False):
        charmap_size = float(self.dataset.get_charmap_size())

        if reverse:
            x = 1.0 / (1 + torch.exp(-x))
            x -= 0.05
            x /= 0.9

            x *= charmap_size
            return x
        else:
            x /= charmap_size

            # logit operation
            x *= 0.9
            x += 0.05
            logit = torch.log(x) - torch.log(1.0 - x)
            log_det = F.softplus(logit) + F.softplus(-logit) + torch.log(torch.tensor(0.9)) \
                      - torch.log(torch.tensor(charmap_size))

            return logit, torch.sum(log_det, dim=1)

    def train_model(self, n_epochs, batch_size, n_samples):
        self.model.train()

        train_losses = []
        matches_at_epoch = {}

        while self.current_epoch < n_epochs:
            self.current_epoch += 1
            self.model.train()

            with tqdm(total=self.dataset.get_train_size()) as bar:
                bar.set_description(f'Epoch {self.current_epoch}')
                batch_loss_history = []

                for b in self.dataset.get_batches(batch_size=batch_size):
                    b = torch.tensor(b).to(self.device).float().contiguous()

                    logit_x, log_det = self.preprocess(b)
                    log_prob = self.model.log_prob(logit_x)
                    log_prob += log_det

                    loss = -torch.mean(log_prob) / float(self.dim)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    batch_loss = float(loss.data)
                    batch_loss_history.append(batch_loss)

                    # Update bar with batch_size progresses and loss value
                    epoch_loss = np.mean(batch_loss_history)
                    bar.set_postfix(loss=epoch_loss)
                    bar.update(b.size(0))

                train_losses.append(epoch_loss)

                if self.current_epoch % self.checkpoint_frequency == 0:
                    self.save(f'checkpoint.epoch{self.current_epoch}.pt')
                    matches_at_epoch[self.current_epoch] = self.evaluate_sampling(n_samples, max_batch_size=10**6)

        return train_losses, matches_at_epoch

    def smoothen_samples(self, samples, uniques):
        for idx, sample in enumerate(samples):
            sample = np.around(samples[idx]).astype(int)
            counter = 0
            noise_d = 0.0
            while ' '.join(map(str, sample)) in uniques:
                if counter > 20:
                    noise_d += 0.05
                    counter = 0
                sample = np.around(samples[idx] + np.random.normal(0.0, self.noise+noise_d, self.dim)).astype(int)
                counter += 1
            samples[idx] = sample
        return samples.astype(int)

    def sample(self, num_samples, uniques=None, save_samples=True):
        with torch.no_grad():
            raw_samples = self.model.sample(num_samples).cpu()
            samples = self.preprocess(raw_samples, reverse=True)
            if uniques is not None and self.noise != 0:
                samples = self.smoothen_samples(samples.cpu().numpy(), uniques)
            else:
                samples = np.around(samples.cpu().numpy()).astype(int)

            if save_samples:
                with open(os.path.join(self.sample_dir, f'samples.epoch{self.current_epoch}.pk'), 'wb') as fw:
                    pickle.dump(samples, fw)

        return samples

    def around_sampling(self, password, num_samples, temperature=0.05):
        self.model.eval()

        pwd = self.dataset.pad_password(tuple(password))
        pwd = np.array([self.dataset.encode_password(pwd)] * num_samples).astype(float)

        x = torch.FloatTensor(pwd).to(self.device)
        with torch.no_grad():
            x, _ = self.preprocess(x)
            z, _ = self.model.flow(x)

            z += torch.distributions.Uniform(low=-temperature, high=temperature).sample(z.shape).to(self.device)

            x = self.model.invert_flow(z)
            x = self.preprocess(x, reverse=True)

            return np.around(x.cpu().numpy()).astype(int)

    def interpolate(self, start, target, steps=50):
        self.model.eval()

        with torch.no_grad():
            start = self.dataset.pad_password(tuple(start))
            target = self.dataset.pad_password(tuple(target))
            start = np.array([self.dataset.encode_password(start)]).astype(float)
            target = np.array([self.dataset.encode_password(target)]).astype(float)

            x1 = torch.FloatTensor(start).to(self.device)
            x2 = torch.FloatTensor(target).to(self.device)

            x1, _ = self.preprocess(x1)
            x2, _ = self.preprocess(x2)

            latents = []

            z1, _ = self.model.flow(x1)
            z2, _ = self.model.flow(x2)

            delta = (z2 - z1) / float(steps)
            latents.append(z1)
            for j in range(1, steps):
                latents.append(z1 + delta * float(j))
            latents.append(z2)

            latents = torch.cat(latents, dim=0)
            logit_results = self.model.invert_flow(latents)
            results = self.preprocess(logit_results, reverse=True)

            return np.around(results.cpu().numpy()).astype(int)

    def evaluate_sampling(self, n_samples, max_batch_size):
        self.model.eval()

        max_batch_size = int(max_batch_size)
        if n_samples < max_batch_size:
            batches, max_batch_size = 1, int(n_samples)
        else:
            batches = math.floor(n_samples / max_batch_size)

        matches = set()
        unique = set()
        with tqdm(range(batches)) as pbar:
            pbar.set_description(desc='Generating sample batch')
            for _ in pbar:
                samples = set(map(tuple, self.sample(max_batch_size, save_samples=False)))
                current_match = samples & self.dataset.test_passwords
                matches.update(current_match)
                unique.update(set(map(lambda x: ' '.join(map(str, x)), samples)))

                pbar.set_postfix({'Matches found': {len(matches)},
                                  'Unique samples': {len(unique)},
                                  'Test set %': ({len(matches) / self.dataset.get_test_size() * 100.0})})

        with open(os.path.join(self.sample_dir, f'matches.epoch{self.current_epoch}.pk'), 'wb') as fw:
            pickle.dump(matches, fw)

        print(f'{len(matches)} matches found ({len(matches) / self.dataset.get_test_size() * 100.0:.4f} of test set).'
              f' out of {len(unique)} unique samples.')

        return matches, unique

    def dynamic_sampling(self, n_samples, max_batch_size, sigma, alpha, gamma):
        self.model.eval()

        matches = set()
        unique = set()
        matched_history = dict()
        batches = math.floor(n_samples / max_batch_size)
        count_samples = 0
        sys.stdout.flush()

        with torch.no_grad():
            with tqdm(range(batches)) as pbar:
                for _ in pbar:
                    sample = self.sample(max_batch_size, save_samples=False)
                    count_samples += len(sample)

                    samples = set(map(tuple, sample))
                    current_match = samples & self.dataset.test_passwords
                    matches.update(current_match)
                    unique.update(set(map(lambda x: ' '.join(map(str, x)), samples)))

                    for match in current_match:
                        if match not in matched_history.keys():
                            matched_history[match] = 0

                    if len(matches) >= alpha:
                        match_set = np.array([match for match in matched_history
                                              if matched_history[match] < gamma])

                        idxs = np.random.randint(0, len(match_set), max_batch_size, np.int32)
                        for match in match_set:
                            matched_history[tuple(match)] += 1

                        x = torch.FloatTensor(match_set).to(self.device)
                        x, _ = self.preprocess(x)
                        matched_z = self.model.flow(x)[0].cpu().numpy()

                        dynamic_mean = matched_z[idxs]
                        dynamic_var = np.full((max_batch_size, self.dim), sigma, dtype=np.float32)
                        self.model.set_prior(dynamic_mean, dynamic_var)

                    pbar.set_postfix({'Num samples': count_samples,
                                      'Matches found': {len(matches)},
                                      'Unique samples': {len(unique)},
                                      'Test set %': ({len(matches) / self.dataset.get_test_size() * 100.0})})

                print(f'{len(matches)} matches found ({len(matches) / self.dataset.get_test_size() * 100.0:.4f}'
                      f' of test set).')
                self.model.reset_prior()

                return matches, unique

    def dynamic_sampling_gs(self, n_samples, max_batch_size, sigma, alpha, gamma, running_mean_len=16):
        self.model.eval()

        matches = set()
        unique = set()
        matched_history = dict()
        batches = math.floor(n_samples / max_batch_size)
        count_samples = 0
        running_matches = np.full(running_mean_len, 20)
        running_idx = 0
        sys.stdout.flush()

        with torch.no_grad():
            with tqdm(range(batches)) as pbar:
                for _ in pbar:
                    sample = self.sample(max_batch_size, uniques=unique, save_samples=False)
                    count_samples += len(sample)

                    samples = set(map(tuple, sample))
                    current_match = samples & self.dataset.test_passwords
                    prev_matches = len(matches)
                    matches.update(current_match)
                    unique.update(set(map(lambda x: ' '.join(map(str, x)), samples)))
                    running_matches[running_idx] = len(matches) - prev_matches

                    for match in current_match:
                        if match not in matched_history.keys():
                            matched_history[match] = 0

                    match_set = np.array([match for match in matched_history
                                          if matched_history[match] < gamma])

                    if len(matches) >= alpha:
                        idxs = np.random.randint(0, len(match_set), max_batch_size, np.int32)
                        for match in match_set:
                            matched_history[tuple(match)] += 1

                        x = torch.FloatTensor(match_set).to(self.device)
                        x, _ = self.preprocess(x)
                        matched_z = self.model.flow(x)[0].cpu().numpy()

                        dynamic_mean = matched_z[idxs]
                        dynamic_var = np.full((max_batch_size, self.dim), sigma, dtype=np.float32)
                        self.model.set_prior(dynamic_mean, dynamic_var)

                    pbar.set_postfix({'Num samples': count_samples,
                                      'Matches found': {len(matches)},
                                      'Unique samples': {len(unique)},
                                      'Running Matches': {np.average(running_matches)},
                                      'Match Set': {len(match_set)},
                                      'Test set %': ({len(matches) / self.dataset.get_test_size() * 100.0})})

                print(f'{len(matches)} matches found ({len(matches) / self.dataset.get_test_size() * 100.0:.4f}'
                      f' of test set).')
                self.model.reset_prior()

                return matches, unique
