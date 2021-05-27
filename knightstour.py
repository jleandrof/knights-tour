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
from math import floor, ceil
from collections import Counter
import time

board_side_len = 8
ch_size = (board_side_len**2)
ch_bit_size = ch_size * 3
pop_size = 60
move_list = [(-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2)]
octal_move_list = ['000', '001', '010', '011', '100', '101', '110', '111']
octal_move_dict = {'000': (-2, -1), '001': (-2, 1), '010': (-1, 2), '011': (1, 2), '100': (2, 1), '101': (2, -1), '110': (1, -2), '111': (-1, -2) }
to_octal = {'000': '0', '001': '1', '010': '2', '011': '3', '100': '4', '101': '5', '110': '6', '111': '7'}
initial_square = (4, 4)
crossover_rate = .8
mutation_rate = .01
alphabet = "01"
gen_size = 1000

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

def gen_chromosome():
    return ''.join(random.choice(octal_move_list) for _ in range(ch_size))

def generate_population():
    return [gen_chromosome() for _ in range(pop_size)]

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

def eval_fitness(chromosome, verb=False):
    board = empty_board()
    x, y = initial_square
    knight = Knight(x, y)
    board[initial_square[0]][initial_square[1]] = 1
    genes = map(''.join, zip(*[iter(chromosome)]*3))

    for index, gene in enumerate(genes):
        if(knight.can_move(board, octal_move_dict[gene])):
            knight.move(octal_move_dict[gene])
            board[knight.x][knight.y] = 1

    if(verb):
        print_board(board)

    total = 0
    for line in board:
        total += sum(line)
    return total

def repair(chromosome):
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

    return ''.join(genes)

def crossover(chromo_a, chromo_b):
    cut_p = floor(ch_bit_size * crossover_rate)
    child_a = chromo_a[:cut_p] + chromo_b[cut_p:]
    child_b = chromo_b[:cut_p] + chromo_a[cut_p:]

    return child_a, child_b

def mutate(chromosome):
    bits = [char for char in chromosome]
    for i in range(len(bits)):
        x = random.random()
        if(x <= mutation_rate):
            bits[i] = alphabet.replace(bits[i], '')

    return ''.join(bits)


def select_top(pop):
    sorted_pop = sorted(pop, key=lambda s: eval_fitness(s), reverse=True)
    return sorted_pop[:floor(pop_size/2)]

def epoch(pop):
    new_gen = selection = select_top(pop)
    slice_point = floor(len(selection)/2)
    sel_1 = selection[:slice_point]
    sel_2 = selection[slice_point:]

    for index in range(len(sel_1)):
        child_a, child_b = crossover(sel_1[index], sel_2[index])
        new_gen.append(repair(mutate(child_a)))
        new_gen.append(repair(mutate(child_b)))

    return new_gen

def progress_bar(i, total):
    i += 1
    p = floor((i*50)/total)
    print('\r[' + ('='*p) + (' '*(50-p)) + '] ' + 'Generations: ' + str(i) + '/' + str(total), end='')

def summary(best, first_tour_gen, tour_counter, time):
    print("Population size: " + str(pop_size))
    print("Crossover rate: " + str(crossover_rate) + "\tMutation rate: " + str(mutation_rate))
    print("Total number of generations: " + str(gen_size))
    print("First tour at generation " + str(first_tour_gen))
    print("Total number of tours found: " + str(tour_counter))
    print("Best solution (octal): " + convert_to_octal(best) + "\tFitness score: " + str(eval_fitness(best)))
    print('Time: %s seconds' % time)

if(__name__ == "__main__"):

    start_time = time.time()
    pop = generate_population()
    pop = [repair(sub) for sub in pop]

    first_tour_gen = None
    tour_counter = 0
    for i in range(gen_size):
        progress_bar(i, gen_size)
        pop = epoch(pop)
        pop = sorted(pop, key=lambda s: eval_fitness(s), reverse=True)
        if(eval_fitness(pop[0]) >= 63):
            if(first_tour_gen is None):
                first_tour_gen = i + 1
            tour_counter += 1

    print()

    pop = [repair(sub) for sub in pop]
    summary(pop[0], first_tour_gen, tour_counter, (time.time() - start_time))
    print()
    eval_fitness(pop[0], True)

    # population_summary(pop, True)
