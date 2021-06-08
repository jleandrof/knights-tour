#! /usr/bin/python
#
# This is an evolutionary algorithm that tries to solve the knight's tour
# problem. It does not care if the solution is a cycle, but allows it to be.
# It tries to repair[1] broken tours whenever possible, adding a bit of
# optimization.
#
# [1] V. S. Gordon and T. J. Slocum, "The knight's tour - evolutionary vs.
# depth-first search,", 2004

import random
import statistics
import plotly.express as px
import pandas as pd
from math import floor, ceil
from collections import Counter
import time
import os
import uuid

image_dir = "___graphs"
if not os.path.exists(image_dir):
    os.mkdir(image_dir)

pop_0_dir = image_dir + "/pop_0"
if not os.path.exists(pop_0_dir):
    os.mkdir(pop_0_dir)

pop_1_dir = image_dir + "/pop_1"
if not os.path.exists(pop_1_dir):
    os.mkdir(pop_1_dir)

pop_2_dir = image_dir + "/pop_2"
if not os.path.exists(pop_2_dir):
    os.mkdir(pop_2_dir)

board_side_len = 8
ch_size = (board_side_len**2)
ch_size_with_params = ch_size + 2
ch_bit_size = ch_size * 3
default_pop_size = 100
move_list = [(-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2)]
octal_move_list = ['000', '001', '010', '011', '100', '101', '110', '111']
octal_move_dict = {'000': (-2, -1), '001': (-2, 1), '010': (-1, 2), '011': (1, 2), '100': (2, 1), '101': (2, -1), '110': (1, -2), '111': (-1, -2) }
to_octal = {'000': '0', '001': '1', '010': '2', '011': '3', '100': '4', '101': '5', '110': '6', '111': '7'}
to_crossover_rate = {'000': .1, '001': .2, '010': .3, '011': .4, '100': .5, '101': .6, '110': .7, '111': .8}
to_mutation_rate = {'000': .01, '001': .011, '010': .012, '011': .013, '100': .014, '101': .015, '110': .016, '111': .017}
initial_square = (4, 4)
default_crossover_rate = .8
default_mutation_rate = .01
alphabet = "01"
default_gen_size = 500
def_iterations = 30

class Knight:
    def __init__(self, xpos, ypos):
        self.x = xpos
        self.y = ypos

    def can_move(self, board, move):
        if(self.x + move[0] not in range(board_side_len)):
            return False
        elif(self.y + move[1] not in range(board_side_len)):
            return False
        elif(board[self.x + move[0]][self.y + move[1]] == 1):
            return False

        return True

    def move(self, move):
        self.x += move[0]
        self.y += move[1]

    def summary(self):
        print("Knight positioned at: " + str((self.x, self.y)))

def empty_board():
    return [[0 for j in range(board_side_len)] for i in range(board_side_len)]

def print_board(board):
    for line in board:
        print(line)

def gen_chromosome(encoded=False):
    size = ch_size
    if(encoded):
        size = ch_size_with_params

    return ''.join(random.choice(octal_move_list) for _ in range(size))

def generate_population(pop_size=default_pop_size, encoded=False):
    return [gen_chromosome(encoded) for _ in range(pop_size)]

def population_summary(pop, bin=False):
    pop = sorted(pop, key=lambda s: eval_fitness(s), reverse=True)
    for chromosome in pop:
        if(bin):
            print(eval_fitness(chromosome), chromosome)
        else:
            print(eval_fitness(chromosome), convert_to_octal(chromosome))

def convert_to_octal(chromosome):
    genes = map(''.join, zip(*[iter(chromosome)]*3))
    result = []
    for gene in genes:
        result.append(to_octal[gene])

    return ''.join(result)

def get_board(chromosome):
    board = empty_board()
    x, y = initial_square
    knight = Knight(x, y)
    board[initial_square[0]][initial_square[1]] = 1
    genes = map(''.join, zip(*[iter(chromosome)]*3))

    for index, gene in enumerate(genes):
        if(knight.can_move(board, octal_move_dict[gene])):
            knight.move(octal_move_dict[gene])
            board[knight.x][knight.y] = 1

    return board

def eval_fitness(chromosome, encoded=False):
    if(encoded):
        chromosome = chromosome[3:-3]

    board = empty_board()
    x, y = initial_square
    knight = Knight(x, y)
    board[initial_square[0]][initial_square[1]] = 1
    genes = map(''.join, zip(*[iter(chromosome)]*3))

    for index, gene in enumerate(genes):
        if(knight.can_move(board, octal_move_dict[gene])):
            knight.move(octal_move_dict[gene])
            board[knight.x][knight.y] = 1

    total = 0
    for line in board:
        total += sum(line)

    return total

def repair(chromosome, encoded=False):
    if(encoded):
        cross_rate = chromosome[:3]
        mut_rate = chromosome[-3:]
        chromosome = chromosome[3:-3]

    board = empty_board()
    x, y = initial_square
    knight = Knight(x, y)
    genes = list(map(''.join, zip(*[iter(chromosome)]*3)))

    for index, gene in enumerate(genes):
        if(knight.can_move(board, octal_move_dict[gene])):
            knight.move(octal_move_dict[gene])
            board[knight.x][knight.y] = 1
        else:
            moves = random.sample(octal_move_list, 8)
            repaired = False
            for move in moves:
                if(knight.can_move(board, octal_move_dict[move])):
                    knight.move(octal_move_dict[move])
                    board[knight.x][knight.y] = 1
                    genes[index] = move
                    repaired = True
                    break

            if(not repaired):
                break

    result = ''.join(genes)
    if(encoded):
        result = cross_rate + result + mut_rate
    return result

def crossover(chromo_a, chromo_b, crossover_rate=default_crossover_rate, encoded=False):
    bit_size = len(chromo_a)
    if(encoded):
        fit_a = eval_fitness(chromo_a, encoded)
        fit_b = eval_fitness(chromo_b, encoded)

        if fit_a >= fit_b:
            crossover_rate = to_crossover_rate[chromo_a[:3]]
        else:
            crossover_rate = to_crossover_rate[chromo_b[:3]]

    cut_p = floor(bit_size * crossover_rate)
    child_a = chromo_a[:cut_p] + chromo_b[cut_p:]
    child_b = chromo_b[:cut_p] + chromo_a[cut_p:]

    return child_a, child_b

def mutate(chromosome, mutation_rate = default_mutation_rate, encoded=False):
    bits = [char for char in chromosome]

    if(encoded):
        mutation_rate = to_mutation_rate[chromosome[-3:]]

    for i in range(len(bits)):
        x = random.random()
        if(x <= mutation_rate):
            bits[i] = alphabet.replace(bits[i], '')

    return ''.join(bits)


def select_top(pop, encoded=False):
    sorted_pop = sorted(pop, key=lambda s: eval_fitness(s, encoded), reverse=True)

    return sorted_pop[:floor(len(pop)/2)]

def epoch(pop, crossover_rate=default_crossover_rate, mutation_rate=default_mutation_rate, encoded=False):
    new_gen = selection = select_top(pop, encoded)

    slice_point = floor(len(selection)/2)
    sel_1 = selection[:slice_point]
    sel_2 = selection[slice_point:]

    for index in range(len(sel_1)):
        child_a, child_b = crossover(sel_1[index], sel_2[index], crossover_rate, encoded)
        new_gen.append(repair(mutate(child_a, mutation_rate, encoded), encoded))
        new_gen.append(repair(mutate(child_b, mutation_rate, encoded), encoded))


    return new_gen

def progress_bar(i, total):
    i += 1
    p = floor((i*50)/total)
    print('\r[' + ('='*p) + (' '*(50-p)) + '] ' + 'Generations: ' + str(i) + '/' + str(total), end='')

def summary(best, first_tour_gen, tour_counter, time):
    """ Only used default values """
    print("Population size: " + str(default_default_pop_size))
    print("Crossover rate: " + str(default_crossover_rate) + "\tMutation rate: " + str(default_mutation_rate))
    print("Total number of generations: " + str(default_gen_size))
    print("First tour at generation " + str(first_tour_gen))
    print("Total number of tours found: " + str(tour_counter))
    print("Best solution (octal): " + convert_to_octal(best) + "\tFitness score: " + str(eval_fitness(best)))
    print('Time: %s seconds' % time)

def average_fitness(pop, encoded=False):
    fitness_list = [eval_fitness(x, encoded) for x in pop]
    return statistics.mean(fitness_list)

def run(pop, gen_size=default_gen_size, image_name="fig", show=False, crossover_rate=default_crossover_rate, mutation_rate=default_mutation_rate, encoded=False):
    first_tour_gen = None
    tour_counter = 0
    av_fits = []
    for i in range(gen_size):
        progress_bar(i, gen_size)
        pop = epoch(pop, crossover_rate, mutation_rate, encoded)
        average_fit = average_fitness(pop, encoded)
        av_fits.append(average_fit)
        pop = sorted(pop, key=lambda s: eval_fitness(s), reverse=True)
        if(eval_fitness(pop[0], encoded) >= 63):
            if(first_tour_gen is None):
                first_tour_gen = i + 1
            tour_counter += 1

    title_text = ''
    if first_tour_gen is None:
        title_text = 'No tours.'
    else:
        title_text = 'First tour at gen {}.'.format(first_tour_gen)

    fig = px.line(av_fits, labels={'index': 'Generations', 'value':'Average fitness'}, title=title_text)

    if(show):
        fig.show()
    else:
        fig.write_image("{}.png".format(image_name))

    print()

def manual():
    pop_0 = generate_population(60)
    pop_0 = [repair(sub) for sub in pop_0]

    pop_1 = generate_population(80)
    pop_1 = [repair(sub) for sub in pop_1]

    pop_2 = generate_population(100)
    pop_2 = [repair(sub) for sub in pop_2]

    cross_rates = [0.8, 0.7, 0.5]
    mut_rates = [0.01, 0.001, 0.005]

    for i in range(def_iterations):
        print("Iteration {}/30".format(i+1))
        run(pop_0, crossover_rate=cross_rates[0], mutation_rate=mut_rates[0], image_name="{}/00_{}".format(pop_0_dir, uuid.uuid1()))
        run(pop_0, crossover_rate=cross_rates[0], mutation_rate=mut_rates[1], image_name="{}/01_{}".format(pop_0_dir, uuid.uuid1()))
        run(pop_0, crossover_rate=cross_rates[0], mutation_rate=mut_rates[2], image_name="{}/02_{}".format(pop_0_dir, uuid.uuid1()))
        run(pop_0, crossover_rate=cross_rates[1], mutation_rate=mut_rates[0], image_name="{}/10_{}".format(pop_0_dir, uuid.uuid1()))
        run(pop_0, crossover_rate=cross_rates[1], mutation_rate=mut_rates[1], image_name="{}/11_{}".format(pop_0_dir, uuid.uuid1()))
        run(pop_0, crossover_rate=cross_rates[1], mutation_rate=mut_rates[2], image_name="{}/12_{}".format(pop_0_dir, uuid.uuid1()))
        run(pop_0, crossover_rate=cross_rates[2], mutation_rate=mut_rates[0], image_name="{}/20_{}".format(pop_0_dir, uuid.uuid1()))
        run(pop_0, crossover_rate=cross_rates[2], mutation_rate=mut_rates[1], image_name="{}/21_{}".format(pop_0_dir, uuid.uuid1()))
        run(pop_0, crossover_rate=cross_rates[2], mutation_rate=mut_rates[2], image_name="{}/22_{}".format(pop_0_dir, uuid.uuid1()))

        run(pop_1, crossover_rate=cross_rates[0], mutation_rate=mut_rates[0], image_name="{}/00_{}".format(pop_1_dir, uuid.uuid1()))
        run(pop_1, crossover_rate=cross_rates[0], mutation_rate=mut_rates[1], image_name="{}/01_{}".format(pop_1_dir, uuid.uuid1()))
        run(pop_1, crossover_rate=cross_rates[0], mutation_rate=mut_rates[2], image_name="{}/02_{}".format(pop_1_dir, uuid.uuid1()))
        run(pop_1, crossover_rate=cross_rates[1], mutation_rate=mut_rates[0], image_name="{}/10_{}".format(pop_1_dir, uuid.uuid1()))
        run(pop_1, crossover_rate=cross_rates[1], mutation_rate=mut_rates[1], image_name="{}/11_{}".format(pop_1_dir, uuid.uuid1()))
        run(pop_1, crossover_rate=cross_rates[1], mutation_rate=mut_rates[2], image_name="{}/12_{}".format(pop_1_dir, uuid.uuid1()))
        run(pop_1, crossover_rate=cross_rates[2], mutation_rate=mut_rates[0], image_name="{}/20_{}".format(pop_1_dir, uuid.uuid1()))
        run(pop_1, crossover_rate=cross_rates[2], mutation_rate=mut_rates[1], image_name="{}/21_{}".format(pop_1_dir, uuid.uuid1()))
        run(pop_1, crossover_rate=cross_rates[2], mutation_rate=mut_rates[2], image_name="{}/22_{}".format(pop_1_dir, uuid.uuid1()))

        run(pop_2, crossover_rate=cross_rates[0], mutation_rate=mut_rates[0], image_name="{}/00_{}".format(pop_2_dir, uuid.uuid1()))
        run(pop_2, crossover_rate=cross_rates[0], mutation_rate=mut_rates[1], image_name="{}/01_{}".format(pop_2_dir, uuid.uuid1()))
        run(pop_2, crossover_rate=cross_rates[0], mutation_rate=mut_rates[2], image_name="{}/02_{}".format(pop_2_dir, uuid.uuid1()))
        run(pop_2, crossover_rate=cross_rates[1], mutation_rate=mut_rates[0], image_name="{}/10_{}".format(pop_2_dir, uuid.uuid1()))
        run(pop_2, crossover_rate=cross_rates[1], mutation_rate=mut_rates[1], image_name="{}/11_{}".format(pop_2_dir, uuid.uuid1()))
        run(pop_2, crossover_rate=cross_rates[1], mutation_rate=mut_rates[2], image_name="{}/12_{}".format(pop_2_dir, uuid.uuid1()))
        run(pop_2, crossover_rate=cross_rates[2], mutation_rate=mut_rates[0], image_name="{}/20_{}".format(pop_2_dir, uuid.uuid1()))
        run(pop_2, crossover_rate=cross_rates[2], mutation_rate=mut_rates[1], image_name="{}/21_{}".format(pop_2_dir, uuid.uuid1()))
        run(pop_2, crossover_rate=cross_rates[2], mutation_rate=mut_rates[2], image_name="{}/22_{}".format(pop_2_dir, uuid.uuid1()))

def stochastic():
    pop = generate_population(encoded=True)
    for i in range(def_iterations):
        run(pop, image_name="{}/STCH_{}".format(image_dir, uuid.uuid1()), encoded=True)

if(__name__ == "__main__"):

    start_time = time.time()
    # manual()
    stochastic()
    print("time: {}".format((time.time() - start_time)))
