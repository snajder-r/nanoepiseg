import os

from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp


class Plotting:

    def plot_met_profile(matrix, samples, segment, site_genomic_pos=None):
        def val_to_color(x, weight=0.5):
            return 1 - np.exp(-np.abs(x) * 0.5)

        sample_color = {'s1': 'g', 's2': 'y', 's3': 'r'}
        y_off = 0
        start = 0
        end = matrix.shape[1]

        # plt.figure(figsize=(12,6))
        for s in ['s3', 's2', 's1']:
            x = np.arange(start, end)
            part_matrix = matrix[:, x][(samples == s)]
            if not site_genomic_pos is None:
                x = site_genomic_pos[x]  # Translate to actual pos on chrom
            active_reads = np.array(
                (part_matrix != 0).sum(axis=1)).flatten() > 0

            part_matrix = part_matrix[active_reads]
            hasval = np.array(part_matrix != 0).flatten()
            y = np.arange(part_matrix.shape[0]) + y_off

            x, y = np.meshgrid(x, y)
            x = x.flatten()[hasval]
            y = y.flatten()[hasval]
            matrix_data = np.array(part_matrix).flatten()[hasval]
            color = [[0, 1, 0, val_to_color(-v)] if v < 0 else [1, 0, 0,
                                                                val_to_color(v)]
                     for v in matrix_data]

            plt.scatter(x, y, c=color, marker='|', s=15)

            x = np.ones(part_matrix.shape[0]) * (x.max() + 20)
            y = np.arange(part_matrix.shape[0]) + y_off
            plt.scatter(x, y, c=sample_color[s])

            y_off += part_matrix.shape[0]

        for i in range(1, len(segment)):
            if segment[i] > segment[i - 1]:
                if not site_genomic_pos is None:
                    i = site_genomic_pos[i]
                plt.plot((i - 1 + 0.5, i - 1 + 0.5), (0, y_off - 1), c='k')

    def plot_segment_profile(matrix, samples, segment, segment_p):
        sample_color = {'s1': 'g', 's2': 'y', 's3': 'r'}
        y_off = 0
        start = 0
        end = matrix.shape[1]

        # plt.figure(figsize=(12,6))
        for s in ['s3', 's2', 's1']:
            x = np.arange(start, end)
            y_ori = np.arange(0, matrix.shape[0])[samples == s]
            part_matrix = matrix[:, x][(samples == s)]
            # x = site_genomic_pos[x] # Translate to actual pos on chrom
            active_reads = np.array(
                (part_matrix != 0).sum(axis=1)).flatten() > 0

            part_matrix = part_matrix[active_reads]
            hasval = np.array(part_matrix != 0).flatten()
            y = np.arange(part_matrix.shape[0]) + y_off

            _, y_ori = np.meshgrid(x, y_ori)
            x, y = np.meshgrid(x, y)
            x = x.flatten()[hasval]
            y = y.flatten()[hasval]
            y_ori = y_ori.flatten()[hasval]
            matrix_data = np.array(part_matrix).flatten()[hasval]

            xs = segment[x]
            color = [np.exp(segment_p[y_ori[i], xs[i]]) for i in range(len(x))]
            color = [np.log(c / (1 - c)) for c in color]

            plt.scatter(x, y, c=color, marker='|', s=15, cmap='RdYlGn_r',
                        vmin=-2.5, vmax=2.5)
            #        if s == 's3':
            #            plt.colorbar()

            x = np.ones(part_matrix.shape[0]) * (x.max() + 20)
            y = np.arange(part_matrix.shape[0]) + y_off
            plt.scatter(x, y, c=sample_color[s])

            y_off += part_matrix.shape[0]

        for i in range(1, len(segment)):
            if segment[i] > segment[i - 1]:
                plt.plot((i - 1 + 0.5, i - 1 + 0.5), (0, y_off), c='k')


class RegionFilter:
    def filter(self, allmet):
        return allmet


class IntervalFilter(RegionFilter):
    def __init__(self, start, end):
        self.start = end
        self.end = end

    def filter(self, allmet):
        return allmet.loc[
            allmet['start'].map(lambda x: self.start <= x < self.end)]


class QuantileFilter(RegionFilter):
    def __init__(self, qfrom, qto):
        self.qfrom = qfrom
        self.qto = qto

    def filter(self, allmet):
        start_locations = list(set(allmet['start']))
        start = np.quantile(start_locations, self.qfrom)
        end = np.quantile(start_locations, self.qto)
        return allmet.loc[allmet['start'].map(lambda x: start <= x < end)]


class MultiIntervalFilter(RegionFilter):
    def __init__(self, ranges):
        self.ranges = ranges

    def filter(self, allmet):
        passed = allmet['start'].map(
            lambda x: any([r[0] <= x < r[1] for r in self.ranges]))
        return allmet.loc[passed]


class MedulloblastomaProject:

    chrom : str
    samples: List[str]
    read_names: List[str]
    read_start: np.array
    read_end: np.array
    matrix_samples: np.array
    allmet: pd.DataFrame
    a: sp.csr_matrix
    a_full: sp.csr_matrix
    met_matrix: sp.csc_matrix
    unique_sites: List[np.int32]
    unique_sites_dict: Dict[np.int32, np.int32]

    def __init__(self, samples=['s1', 's2', 's3']):
        self.samples = samples

    def load_pickled_metcalls(self, metcall_dir: str, chrom: str,
                              region_filter: RegionFilter = RegionFilter(),
                              filter_bad_reads: bool = False):
        self.chrom = chrom
        sample_met = dict()
        rn_sample_dict = dict()
        for s in self.samples:
            sample_file = os.path.join(metcall_dir,
                                       '%s_%s_met_cpg.pkl' % (s, chrom))
            sample_met[s] = pd.read_pickle(sample_file, compression='gzip')
            rn_sample_dict = {**rn_sample_dict, **{r: s for r in list(
                set(sample_met[s]['read_name']))}}
            print(len(np.unique(sample_met[s]['read_name'])))
        allmet = pd.concat((sample_met[s] for s in self.samples))
        # Apply filter
        allmet = region_filter.filter(allmet)

        # Filtering regions with too high or too low coverage
        # High coverage spikes are likely highly repeated regions
        if filter_bad_reads:
            while True:
                # Filter regions that have insanely high coverage
                loc_coverage = allmet[['start', 'read_name']].groupby(
                    'start').count()
                loc_coverage_mean = np.mean(loc_coverage['read_name'])
                loc_coverage_std = np.std(loc_coverage['read_name'])
                cov_spike = loc_coverage_mean + loc_coverage_std * 4

                bad_locs = set(
                    loc_coverage.index[(loc_coverage['read_name'] > cov_spike)])
                allmet = allmet[
                    allmet['start'].map(lambda x: not x in bad_locs)]

                # Filter reads that have really low number of sites
                read_coverage = allmet[['start', 'read_name']].groupby(
                    'read_name').count()
                bad_reads = set(
                    read_coverage.index[read_coverage['start'] < 10])
                allmet = allmet[
                    allmet['read_name'].map(lambda x: not x in bad_reads)]

                if len(bad_locs) == 0 and len(bad_reads) == 0:
                    break

        # Detect where reads start and end, and then create sorted list of reads

        allmet = allmet.sort_values(['read_name', 'start'])
        read_start = allmet[['read_name', 'start']].groupby(
            'read_name').min().start
        read_end = allmet[['read_name', 'start']].groupby(
            'read_name').max().start
        read_start = read_start.sort_values()
        read_end = read_end[read_start.index]
        read_names = np.array(read_start.index)

        matrix_samples = [rn_sample_dict[r] for r in list(read_names)]

        self.read_names = read_names
        self.read_start = np.array(read_start)
        self.read_end = np.array(read_end)
        self.matrix_samples = np.array(matrix_samples)
        self.allmet = allmet

    def build_read_adjacency_matrix(self):
        # Building read vs read adjacency matrix
        bs = 10000
        a_chunk = np.zeros((bs, len(self.read_start)), dtype='bool')
        a = None
        for i in range(len(self.read_start)):
            a_chunk[i % bs, i:] = self.read_start[i:] <= self.read_end[i]
            if i % 1000 == 0:
                print('{0:.2f}'.format(i / len(self.read_start) * 100), end=',')
            if ((i + 1) % bs == 0) or i == (len(self.read_start) - 1):
                if a is None:
                    print('\nBuilding csr_matrix from chunk')
                    a = sp.csr_matrix(a_chunk)
                else:
                    print('\nBuilding csr_matrix from chunk and stacking')
                    a = sp.vstack((a, sp.csr_matrix(a_chunk)))
                a_chunk[:, :] = 0

        # A is triangular
        a = sp.csr_matrix(a)[:a.shape[1], ]
        self.a = a
        # A_full has both triangles
        a_full = a + a.T - sp.diags(a.diagonal(), dtype='bool')
        self.a_full = a_full

    def build_sparse_met_matrix(self):
        unique_sites = np.sort(list(set(self.allmet.start)))
        unique_sites_dict = {unique_sites[i]: i for i in
                             range(len(unique_sites))}
        # Building sparse read vs site methylation matrix
        # As a compromise between memory usage and processing speed, 
        # we build dense blocks (fast, but takes memory), 
        # then make them sparse (slow, but memory efficient), 
        # and then concatenate the sparse blocks
        met_matrix = sp.lil_matrix((len(self.read_names), len(unique_sites)))
        read_dict = {self.read_names[i]: i for i in range(len(self.read_names))}
        cur_rn = ''
        i = 0
        for e in self.allmet.itertuples():
            if i % 1000000 == 0:
                print('{0:.2f}'.format(i / self.allmet.shape[0] * 100), end=',')
            if i + 1 % 100000000 == 0:
                print()
            i += 1

            if cur_rn != e[5]:
                cur_rn = e[5]
                read_idx = read_dict[cur_rn]
            met_matrix[read_idx, unique_sites_dict[e[3]]] = e[6]
        self.met_matrix = sp.csc_matrix(met_matrix)
        self.unique_sites = unique_sites
        self.unique_sites_dict = unique_sites_dict

    def activation_function(x, weight=0.5):
        return (1 - np.exp(-np.abs(x) * 0.5))

    def sparse_divide_nonzero(a, b):
        inv_b = b.copy()
        inv_b.data = 1 / inv_b.data
        return a.multiply(inv_b)

    def compute_distance_matrix(self, l_met_matrix=None,
                                weigh_by_site_heterogeniety=False, min_cov=30,
                                activation_function=activation_function):

        total_met_rate = 0
        if l_met_matrix is None:
            l_met_matrix = self.met_matrix

        has_info = (l_met_matrix != 0).astype('uint32')
        overlap_size = has_info * has_info.T
        overlap_size = overlap_size - sp.diags(overlap_size.diagonal(),
                                               dtype='uint32')

        A_full = overlap_size > min_cov

        met_matrix_met = sp.csr_matrix(l_met_matrix > 1, dtype='float')
        met_matrix_unmet = sp.csr_matrix(l_met_matrix < -1, dtype='float')

        met_matrix_met = met_matrix_met.multiply(l_met_matrix)
        met_matrix_met.data = activation_function(met_matrix_met.data,
                                                  weight=1 - total_met_rate)
        met_matrix_unmet = met_matrix_unmet.multiply(l_met_matrix)
        met_matrix_unmet.data = activation_function(-met_matrix_unmet.data,
                                                    weight=total_met_rate)

        if weigh_by_site_heterogeniety:
            site_heterogeniety = 1 - (np.abs(met_matrix_met.sum(axis=0) - \
                                             met_matrix_unmet.sum(axis=0))) / (
                                             met_matrix_met.sum(
                                                 axis=0) + met_matrix_unmet.sum(
                                         axis=0))
            met_matrix_met = met_matrix_met.multiply(site_heterogeniety)
            met_matrix_unmet = met_matrix_unmet.multiply(site_heterogeniety)

        both_met = met_matrix_met * met_matrix_met.T
        both_unmet = met_matrix_unmet * met_matrix_unmet.T

        agreement = (both_met + both_unmet).multiply(A_full)
        disagreement = (
                    met_matrix_unmet * met_matrix_met.T + met_matrix_met * met_matrix_unmet.T).multiply(
            A_full)

        D = sparse_divide_nonzero(agreement, agreement + disagreement)
        return D
