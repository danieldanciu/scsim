import argparse
import os
import time
import sys
import logging

from Bio import SeqIO
import numpy as np
import scipy.stats as ss
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MAX_ITER = 20
RES_DIR = ''


def read_fasta(input_file, filter_ids, start=0, end=1e12):
    """ Read a fasta file.
      Parameters:
          input_file (string) input FASTA file name
          start (int) for each sequence in the fasta file, only consider positions after start
          end (int) for each sequence in the fasta file, only consider positions until end
          ids (list(string)) only consider sequences with an id present in ids
      Returns:
          dict(string, string) containing the sequence for each id
      """

    seqs = []
    seq_ids = []
    # convert fasta to character string with BioPython
    for seq in SeqIO.parse(input_file, 'fasta'):
        if seq.id not in filter_ids:
            continue
        if len(seq.seq) < 300:
            logger.warning(f'Sequence {seq.id} too short. Ignoring.')
            continue
        logger.debug(f'Id:  {seq.id} Size: {len(seq.seq)}')
        seqs.append(seq.seq[start:min(end, len(seq.seq))])
        seq_ids.append(seq.id)

    return seqs, seq_ids


def write_snv_mtx(ids, snv_matrices, snv_locs):
    """ Function to store SNV flags matrix in csv and LaTeX formats """

    for i in range(len(ids)):
        snv_mtx = snv_matrices[i]
        snv_loc = snv_locs[i]

        (n_snv, n_sc) = snv_mtx.shape

        if n_snv == 0 or n_sc == 0:
            return

        # cast flags matrix as pandas dataframe
        snv_df = pd.DataFrame(snv_mtx)

        # rename dataframe rows and columns
        snv_df.columns = ['PROTO%d' % i for i in range(1, n_sc + 1)]
        snv_df.index = snv_loc + 1

        # write data frame to markdown
        snv_df.to_csv(path_or_buf=os.path.join(RES_DIR, f'snv_flags_{ids[i]}.md'), sep='|', index=True, header=True,
                      mode='w', float_format='%.2g')

        # export data frame to latex
        snv_df.to_latex(os.path.join(RES_DIR, f'snv_flags_{ids[i]}.tex'))


def write_fasta(seqs, ids, out_file):
    """ Writes a map of id->sequence pairs to a FASTA file """

    # write header and sequence to file
    with open(out_file, 'w') as fasta_f:
        for seq_id, seq in zip(ids, seqs):
            fasta_f.write('>' + seq_id + '\n' + ''.join(seq) + '\n')
        fasta_f.flush()
        logger.debug(f'Wrote {out_file}')


def write_sc(ids, sc_obj, seq_type='wga_gtype', out='output'):
    """ Write a sequence to two fasta files, one for each allele"""

    # extract sequence for each allele
    seqs_a1 = []
    seqs_a2 = []
    for idx in range(len(ids)):
        seq_a1 = []
        seq_a2 = []
        for i in range(len(sc_obj[idx])):
            # extract gnotype as list
            gtype = list(sc_obj[idx][i][seq_type])
            seq_a1.append(gtype[0])
            # loop through loci
            seq_a2.append(gtype[1])
        seqs_a1.append(seq_a1)
        seqs_a2.append(seq_a2)
    # write to two fasta files, one for each allele
    write_fasta(seqs_a1, ids, out + '_a1')
    write_fasta(seqs_a2, ids, out + '_a2')


def apply_ado(src_gt):
    """ Utility function to apply Allelic Dropout on a given genotype.
        ADO is defined as the random non-amplification of one of the alleles present in a heterozygous sample.
    """
    return 2 * src_gt[np.random.choice(2)]


def apply_fp(src_gt):
    """Utility function to apply false positives on a given genotype.
       A false positive occurs at a homzygous loci that may seem heterozygous after amplification
    """

    tmp = list(src_gt)
    fp_gt = None
    # assert homozygous genotype
    if tmp[0] == tmp[1]:
        if tmp[0] == 'A':
            fp_gt = 'T' + tmp[1]
        elif tmp[0] == 'T':
            fp_gt = 'A' + tmp[1]
        elif tmp[0] == 'G':
            fp_gt = 'C' + tmp[1]
        elif tmp[0] == 'C':
            fp_gt = 'G' + tmp[1]
        else:
            logger.info('Unrecognized genotype.')
    else:
        fp_gt = src_gt

    return fp_gt


def gen_snv_transition_matrix(ts_tv_p=0.71,
                              ts_hm_ht_p=0.5,
                              tv_l_p=0.5,
                              tv_hm_ht_p=0.8):
    """ Function to compute SNV matrix of transition probabilities,
    default rates follow SInC reference (Fig1.a) """

    G_ref = {'AA': 0, 'CC': 1, 'GG': 2, 'TT': 3}
    G_alt = {0: 'AA', 1: 'AC', 2: 'AG', 3: 'AT', 4: 'CC', 5: 'CG', 6: 'CT', 7: 'GG', 8: 'GT', 9: 'TT'}

    # initialize matrices of transition probabilities
    T_transition = np.zeros((len(G_ref), len(G_alt)))
    T_transversion_r = np.zeros((len(G_ref), len(G_alt)))
    T_transversion_l = np.zeros((len(G_ref), len(G_alt)))

    # fill in transition probabilties with specified rates
    T_transition[0, 7] = T_transition[1, 9] = T_transition[2, 0] = T_transition[3, 4] = ts_hm_ht_p
    T_transition[0, 2] = T_transition[1, 6] = T_transition[2, 2] = T_transition[3, 6] = 1 - ts_hm_ht_p

    T_transversion_l[0, 4] = T_transversion_l[1, 0] = T_transversion_l[2, 9] = T_transversion_l[3, 7] = tv_hm_ht_p
    T_transversion_l[0, 1] = T_transversion_l[1, 1] = T_transversion_l[2, 8] = T_transversion_l[3, 8] = 1 - tv_hm_ht_p

    T_transversion_r[0, 9] = T_transversion_r[1, 7] = T_transversion_r[2, 4] = T_transversion_r[3, 0] = tv_hm_ht_p
    T_transversion_r[0, 3] = T_transversion_r[1, 5] = T_transversion_r[2, 5] = T_transversion_r[3, 3] = 1 - tv_hm_ht_p

    # aggregate intermediate matrices
    T_transversion = tv_l_p * T_transversion_l + (1 - tv_l_p) * T_transversion_r
    T_snv = ts_tv_p * T_transition + (1 - ts_tv_p) * T_transversion

    # assert final matrix rows sum to 1
    if np.sum(T_snv) != len(G_ref):
        raise ValueError('Transition probabilities matrix rows do not sum to 1!')

    return T_snv


def sim_alt_gtype(ref, T_snv):
    """ Function to simulate alternate SNV genotype using a matrix of
    transition probabilities """

    G_ref = {'AA': 0, 'CC': 1, 'GG': 2, 'TT': 3}
    G_alt = {0: 'AA', 1: 'AC', 2: 'AG', 3: 'AT', 4: 'CC', 5: 'CG', 6: 'CT', 7: 'GG', 8: 'GT', 9: 'TT'}

    ref = ref.upper()
    if ref not in G_ref:  # probably an NN
        return ref

    # get row of probabilities corresponding to reference input
    ref_p = T_snv[G_ref[ref], :]
    # sample alternate genotype from reference probability vector
    alt_sample = ss.multinomial.rvs(1, p=ref_p)
    alt_idx = np.where(alt_sample == 1)[0][0]  # get sample index
    alt_gtype = G_alt[alt_idx]  # convert index to genotype

    return alt_gtype


def simulate_snv(source, snv_loc, alt_gtype, ado_rate=0.2, fp_rate=3.2e-5):
    """ Simulate all SNVs in given locations with specified rates """

    n_snv = len(snv_loc)
    N = len(source)

    sc_gtypes = [dict(loc=i,
                      ref_gtype=2 * source[i],
                      isSNV=False,
                      alt_gtype=2 * source[i],
                      isADO=False,
                      ado_gtype=2 * source[i],
                      isFP=False,
                      fp_gtype=2 * source[i],
                      wga_gtype=2 * source[i])
                 for i in range(N)]

    # Apply SNVs
    for i in snv_loc:
        sc_gtypes[i]['isSNV'] = True
        sc_gtypes[i]['alt_gtype'] = alt_gtype[i]
        sc_gtypes[i]['ado_gtype'] = sc_gtypes[i]['alt_gtype']
        sc_gtypes[i]['fp_gtype'] = sc_gtypes[i]['ado_gtype']
        sc_gtypes[i]['wga_gtype'] = sc_gtypes[i]['fp_gtype']

    # Apply ADOs
    ado_loc = np.random.choice(N, int(ado_rate * N))
    for i in ado_loc:
        sc_gtypes[i]['isADO'] = True
        sc_gtypes[i]['ado_gtype'] = apply_ado(sc_gtypes[i]['alt_gtype'])
        sc_gtypes[i]['fp_gtype'] = sc_gtypes[i]['ado_gtype']
        sc_gtypes[i]['wga_gtype'] = sc_gtypes[i]['fp_gtype']

    # Apply FPs
    fp_loc = np.random.choice(N, int(fp_rate * N))
    for i in fp_loc:
        sc_gtypes[i]['isFP'] = True
        sc_gtypes[i]['fp_gtype'] = apply_fp(sc_gtypes[i]['ado_gtype'])
        sc_gtypes[i]['wga_gtype'] = sc_gtypes[i]['fp_gtype']

    return sc_gtypes


def get_snv_locations(cell_count, num_snv, num_groups):
    """ Creates a boolean matrix that indicates which snv positions are assigned to which cells.
    Parameters:
        cell_count (int) total number of cells
        num_snv (int) number of single nucleotide variation positions
        num_groups (int) number of cell groups (the sum of cells in each group amount to cell_count)
    Returns:
        tuple of:
          - matrix of size (num_snv, num_groups) containing the variations for each cell group
          - dict of int: float map of snv locus to proportion of shared SNVs across all single cells for that locus
    """

    # initialize SNV flags matrix
    snv_mtx = np.zeros((num_snv, num_groups), dtype=bool)

    # prepare indices for three simulation scenarios
    row_idx = np.linspace(start=0, stop=num_snv, num=4).astype(int)
    col_idx = int(cell_count / 2)

    # scenario (1): all prototypes (cell types)4 share one third of the SNV locations
    snv_mtx[:row_idx[1], ] = True

    # scenario (2): half of the prototypes share one third of the SNV locations
    snv_mtx[row_idx[1]:row_idx[2], col_idx:] = True

    # scenario (3): simulate shared and singleton SNVs uniformly
    p = {}  # initialize SNV simulation proportions
    for i in range(row_idx[2], row_idx[3]):
        # simulate and store proportion of shared SNVs across all
        # single cells in row i
        p[i] = ss.uniform.rvs(0, 1)
        # simulate SNV locations (encoded as '1's) for all single cells
        # in row i, based on simulated proportion p_i
        snv_mtx[i] = ss.binom.rvs(1, p[i], size=num_groups) == 1

    return snv_mtx, p


def simulate_sc(sources, ids, n_snv, cell_counts, out='', ado_rate=0.2, fp_rate=3.2e-5):
    """ Generates fasta files, each containing the simulated reference sequence for a given single cell allele.
        Parameters:
          sources (list(str)): reference sequence used to simulate single cells.
          ids (list(str)): ids for each of the strings in sources (e.g. chromosome names)
          n_snv (int): total number of SNVs to simulate per cell.
          cell_counts (list): number of single cells to generate for each of the n_sc mutant references.
          snv_loc (int list): optional list of SNV locations to simulate.
          out (string): output file name, default is ''.
          ado_rate (float): allelic dropout rate, must be in [0,1].
          fp_rate (float): false positive rate, must be in [0,1].
    """

    if not sources:
        logger.info("Empty source strings. Nothing to do")
        return dict(sc_fnames=[], snv_mtx=np.empty((0, 0)), snv_loc=[])

    # total number of single cells
    n_sc = sum(cell_counts)
    n_proto = len(cell_counts)

    # sample snv locations across source
    # NB: avoid extremities of the sequence
    offset = 100
    total_length = sum(len(s) for s in sources)
    proportions = [len(s) / total_length for s in sources]
    num_snvs = [round(p * n_snv) for p in proportions]
    snv_locs = [np.linspace(start=offset, stop=len(s) - offset, num=num_snv).astype(int) for
                s, num_snv in zip(sources, num_snvs)]

    snv_matrices = []
    p_lists = []
    print('Sources is ', sources)
    print('Num snvs is ', num_snvs)
    # generate the matrix that decides which cells will get which SNVs
    for n_snv in num_snvs:
        (snv_mtx, p_lst) = get_snv_locations(n_sc, n_snv, n_proto)
        snv_matrices.append(snv_mtx)
        p_lists.append(p_lst)

    # generate alternate genotypes for all snv locations
    T_snv = gen_snv_transition_matrix()
    alt_gtypes = [{loc: sim_alt_gtype(ref=2 * source[loc], T_snv=T_snv)
                   for loc in snv_loc} for source, snv_loc in zip(sources, snv_locs)]

    # loop through single cells
    start_time = time.perf_counter()  # start timer
    sc_fnames = []
    r = 1
    for group in range(len(cell_counts)):

        logger.info('Simulating prototype {} with {} single cells ...'.format(group + 1, cell_counts[group]))

        sc_gtypes = []
        for idx in range(len(sources)):
            snv_mtx = snv_matrices[idx]
            sc_gtypes.append(
                simulate_snv(sources[idx], snv_locs[idx][snv_mtx[:, group]], alt_gtypes[idx], ado_rate, fp_rate))

        for cell_idx in range(cell_counts[group]):
            logger.info('Single cell simulation {} done! Writing sequence to fasta file ...'.format(cell_idx + 1))

            # write simulated single cell sequence to fasta
            outname = out + '%d' % (r)
            sc_fnames.append(outname)  # store file names
            write_sc(ids, sc_gtypes, 'wga_gtype', 'sc' + outname)  # write sequence after WGA

            # export SNV locations to bed files
            # TODO: only locations for the first sequence/chromosome are exported for now
            bed_df = pd.DataFrame({'chrom': 'ref_source',
                                   'chromStart': snv_locs[0][snv_matrices[0][:, group]],
                                   'chromEnd': snv_locs[0][snv_matrices[0][:, group]] + 1})
            bed_df.to_csv(path_or_buf=os.path.join(args.o, f'snv_sc{r}.bed'), sep='\t', header=False, mode='w',
                          index=False)
            r += 1
        write_sc(ids, sc_gtypes, 'alt_gtype', f'prototype{out}{group + 1}')  # write sequence before WGA

    sim_time = time.perf_counter() - start_time  # store simulation time for display
    logger.info(
        f'All done! Simulated {n_snv} SNVs in {len(cell_counts)} prototypes and {n_sc} single cells in {sim_time:.2f} '
        f'seconds.')

    return sc_fnames, snv_matrices, snv_locs


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument('--num_mutant_cells', help='Number of single cells for each of the --num_mutant_types type',
                      nargs='+', default=[10])
    args.add_argument('--ids', help='The ids in the reference fasta file for which to generate simulated data.',
                      nargs='+',
                      default=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                               '17', '18', '19', '20', '21', '22', 'X', 'Y'])
    args.add_argument('--snv', default=1000, help='Number of single nucleotide variants (SNVs)')
    args.add_argument('--reference', default=None,
                      help='Fasta file containing the reference genome for which the mutated variants are generated')
    args.add_argument('--seed', default=1096, help='Random seed for simulation reproducibility')
    args.add_argument('--start', default=0, type=int,
                      help='Start of the region that is considered for variation')
    args.add_argument('--stop', default=1e12, type=int,
                      help='End of the region that is considered for variation, '
                           'if not specified the entire region is considered')
    args.add_argument('--prefix', help='Prefix for the generated single cell fasta files', default='')
    args.add_argument('--o', help='Output directory', default='./')
    args.add_argument('--ado', help='Allelic dropout rate', default='0.2', type=float)
    args.add_argument('--fp', help='False positive rate', default='3.2e-5', type=float)
    args = args.parse_args()

    if args.stop - args.start < 300:
        logger.exception('Difference between start and stop must be > 300')
        exit(1)

    np.random.seed(args.seed)  # set environment random seed

    # read and filter source to target size
    source_strings, ids = read_fasta(args.reference, args.ids, args.start, args.stop)
    # write filtered source to FASTA
    write_fasta(source_strings, ids, os.path.join(args.o, 'reference.fa'))

    # get number of cells to simulate in current experiment
    num_mutant_cells = [int(n) for n in args.num_mutant_cells]
    num_cells = sum(num_mutant_cells)
    num_groups = len(num_mutant_cells)
    logger.info(f'Starting Experiment: simulating {num_groups} prototypes and {num_cells} single cell sequences...')

    # simulate prototypes and single cells and write to fasta files (one per allele)
    fnames, snv_matrices, snv_locs = simulate_sc(source_strings, ids, args.snv, num_mutant_cells, args.prefix,
                                                    args.ado, args.fp)

    # write SNV flags matrix to file
    write_snv_mtx(ids, snv_matrices, snv_locs)

    # write filenames to text file
    ffnames = open(os.path.join(RES_DIR, 'ref_fnames.txt'), 'w')
    [ffnames.write('after_' + str(i) + '_a1.fa\n' + 'after_' + str(i) + '_a2.fa\n') for i in fnames]
    [ffnames.write('prototype_' + str(i) + '_a1.fa\n' + 'prototype_' + str(i) + '_a2.fa\n') for i in fnames]
    ffnames.write('ref_source.fa\n')
    ffnames.close()
