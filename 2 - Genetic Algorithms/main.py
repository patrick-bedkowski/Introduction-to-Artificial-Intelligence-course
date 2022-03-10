import pandas as pd
import random
import copy

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import datetime
import time
from typing import List

# Custom palette for graph with 25 different colors
palette_data_science_25 = [
        '#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54',
        '#ffa600', '#152024', '#004243', '#006449', '#438535',
        '#989d03', '#f9db11', '#6bc9ff', '#749bd2', '#71709f',
        '#30c9c7', '#45293c', '#e1da36', '#cd17ff', '#ff5036',
        '#c75413', '#0084ff', '#b3747b', '#7a8cff', '#ff5757'
]

class Graph:
    """
    Representation of a graph.
    Holds nodes and vertices.
    """
    def __init__(self) -> None:
        self.vertices = []
        self.edges = []
        self._type_bipartite = False
        self.U = []
        self.V = []

    def _append_edges(self, edges: List[any]) -> None:
        """
        Appends vertices and edges to graph.
        :param edges: Edges
        :return:
        """
        # append edges
        for edge in edges:
            self.edges.append(edge)

    def _complete(self, N: int) -> None:
        """
        Create complete graph. Appends vertices and edges to graph.
        :param N: Number of vertices
        :return: None
        """
        edges = []
        for vertex_1 in range(1, N + 1):  # iterate through vertices
            self.vertices.append(vertex_1)  # add vertex
            for vertex_2 in range(1, N + 1):  # iterate through vertices
                vertices_pair = (vertex_1, vertex_2)
                if vertex_1 != vertex_2 and (vertex_2, vertex_1) not in edges:  # cannot connect the same vertex with itself
                    edges.append(vertices_pair)

        # append edges
        self._append_edges(edges)

    def _bipartite(self, N: int) -> None:
        """
        Create bipartite graph. Each vertex is connected at least to
        two others vertices. Appends vertices and edges to graph.
        :param N: Number of vertices
        :return: None
        """

        # divide edges into two groups #
        split_value = random.randint(5, N-4)  # choose split value

        edges = []
        for vertex_1 in range(1, N + 1):  # iterate through vertices
            self.vertices.append(vertex_1)
            if vertex_1 <= split_value:
                self.U.append(vertex_1)
                for _ in range(0, 2):  # append two randomly chosen vertices to the vertex
                    random_vertex = random.randint(split_value+1, N)
                    vertices_pair = (vertex_1, random_vertex)
                    if vertex_1 != random_vertex and vertices_pair not in edges and (random_vertex, vertex_1) not in edges:
                        edges.append(vertices_pair)
            else:
                self.V.append(vertex_1)
                for _ in range(0, 2):  # append two randomly chosen vertices to the vertex
                    random_vertex = random.randint(1, split_value-1)
                    vertices_pair = (vertex_1, random_vertex)
                    if vertex_1 != random_vertex and vertices_pair not in edges and (random_vertex, vertex_1) not in edges:
                        edges.append(vertices_pair)

        # append edges
        self._append_edges(edges)

    def _random(self, N: int) -> None:
        """
        Create graph that has 60% connections of a complete graph.
        Appends vertices and edges to graph.
        :param N: Number of vertices
        :return: None
        """

        for vertex in range(1, N + 1):  # add N vertices
            self.vertices.append(vertex)

        # number of edges need to be 60% of complete graph

        # create edges connections
        edges = []
        while (len(edges) / ((N ** 2 - N) / 2)) <= 0.6:
            r_vertex_one = random.randint(1, N)  # choose random vertex to form a pair
            r_vertex_two = random.randint(1, N)  # choose random vertex to form a pair

            edge = (r_vertex_one, r_vertex_two)

            if r_vertex_one != r_vertex_two and edge not in edges and edge[::-1] not in edges:
                edges.append(edge)

        # append edges
        self._append_edges(edges)

    def _random_groups(self, N: int) -> None:
        """
        Create graph with random groups. Firstly 15% of vertices are randomly
        chosen to be connected with 50% of vertices. The rest 85% has one
        connection with random vertices.
        Appends vertices and edges to graph.
        :param N: Number of vertices
        :return: None
        """

        for vertex in range(1, N + 1):  # add N vertices
            self.vertices.append(vertex)

        group_vertices_15 = []  # 15% of all available vertices
        while len(group_vertices_15) < (N*0.15):
            rand_vertex = random.randint(1, N)
            if rand_vertex not in group_vertices_15:
                group_vertices_15.append(rand_vertex)

        group_vertices_90 = set(self.vertices) - set(group_vertices_15)  # 85% of the left vertices

        # connect 15% of vertices to the 70% of vertices
        group_edges_15 = []
        for vertex_15 in group_vertices_15:
            vertex_edges = []
            while len(vertex_edges) < int(N*0.5):
                rand_vertex = random.randint(1, N)
                edge = (vertex_15, rand_vertex)
                if vertex_15 != rand_vertex and edge not in group_edges_15 and edge[::-1] not in group_edges_15:
                    vertex_edges.append(edge)
            group_edges_15.extend(vertex_edges)

        # connect the rest 85% of vertices
        group_edges_85 = []
        for vertex_85 in group_vertices_90:
            rand_vertex = random.randint(1, N)
            edge = (vertex_85, rand_vertex)
            if vertex_85 != rand_vertex and edge not in group_edges_85 and edge[::-1] not in group_edges_85:
                group_edges_85.append(edge)

        all_edges = group_edges_85 + group_edges_15

        # append edges
        self._append_edges(all_edges)

class GraphColorizer:
    """
    Implementation of evolutionary algorithm used to solve the NP-complete
    problem of graph coloring.

    Parameters
    ----------
    max_iterations: int
        Maximum number of iterations after which the algorithm is stopped.
    population_size: int
        Number of individuals in the population.
    mutation_probability : float
        Probability of mutation occurring in the individual's genes.
    gene_mutation_probability: float
        Probability of single gene mutation for the individual.
    graph_type: str
        Type of graph. One of {'complete', 'bipartite', 'random', 'random groups'}
    max_no_improvements: int, default=None
        Maximum number of iterations that fit the condition of a new
        generation of population having worse best individual
        (worse fitness score) than the current best individual.
        Iterations stops if the number of such occurrences exceeds this
        parameter.
    """
    def __init__(self,
                 max_iterations,
                 population_size,
                 mutation_probability,
                 gene_mutation_probability,
                 graph_type,
                 max_no_improvements = None
                 ):

        self.graph = self._load_graph(graph_type)
        self.graph_type = graph_type
        self.T = max_iterations
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.gene_mutation_probability = gene_mutation_probability
        self.max_no_improvements = self.T if max_no_improvements is None else max_no_improvements
        self.number_of_vertices = len(self.graph.vertices)
        self.number_of_edges = len(self.graph.edges)

    def run_algorithm(self):
        """
        Main method with implementation of evolutionary algorithm to solve
        graph coloring problem.
        Implemented stop conditions:
        - achieved limit of iterations
        - no improvements in last n number of iterations
        (if n has not been passed to class, it is equal to maximum number
        of iterations)
        :return: List containing population with fitness scores, best achieved fitness,
        best individual, number of iterations, quantity of colors that algorithm started with
        """
        t = 0  # counter of iterations
        no_impvt_counter = 0  # counter of no improvements
        population_scores = []

        population, starting_colors = self._initialize()  # initialize population
        n_starting_colors = len(set(starting_colors))  # number of colors at the beginning

        print('\nPopulation has been initialized...')
        print(f'\nStarting graph has {n_starting_colors} unique colors')

        print('\nFinding suitable solution...')
        best_individual, best_fitness = self._best(population)  # find the best individual

        population_scores.append((best_individual, best_fitness))

        while t < self.T and no_impvt_counter <= self.max_no_improvements:
            population = self._selection(population)
            population = self._mutation(population, False)

            individual, fitness = self._best(population)
            if fitness > best_fitness:
                no_impvt_counter = 0
                best_fitness = fitness
                best_individual = individual
            else:
                no_impvt_counter += 1

            pop_score = (individual, fitness)
            population_scores = copy.deepcopy(population_scores)
            population_scores.append(pop_score)

            t += 1

        return population_scores, best_fitness, best_individual, t, n_starting_colors

    def _initialize(self):
        """Create the initial population. We randomly choose as many
        individuals as is set in self.population_size.
        Create population. Each individual consists of sequence of integers, where a
        sequence position 1, 2, ..., n represents the number of vertex, and the value
        (integer) of the sequence on the position n represents the color of the vertex n
        The initial population consists only of valid individuals
        (connected vertices of different colors).
        """
        vertices = self.graph.vertices
        colors = [c for c in range(1, len(vertices)+1)]  # define available colors
        population = []

        for _ in range(self.population_size):
            population.append(colors)  # append population

        return population, colors

    def _is_valid(self, sequence) -> bool:
        """
        Check if generated vertices and colors are valid. Meaning that two connected
        vertices need to be of different colors.
        """
        results = []  # list containing scores for pair's color
        for w1, w2 in self.graph.edges:  # iterate through pairs of vertices
            is_valid = sequence[w1-1] != sequence[w2-1]
            results.append(is_valid)
        return all(results)  # return True if all elements are True

    def _fitnesses(self, population):
        """
        Calculates fitnesses for every individual in the population.
        :param population: List with individuals
        :return: list of tuples in form of (individual, fitness(individual)).
        """
        return [(individual, self._fitness_score(individual)) for individual in population]

    def _fitness_score(self, individual):
        """
        Fitness functions calculating a number of colors used. We want to maximize that.
        """
        number_of_unique_colors = len(set([c for c in individual]))
        return 100 * self.number_of_vertices/number_of_unique_colors

    def _normalize_fitness(self, population):
        """
        Used to normalize fitness of individuals.
        :param population: population with fitness scores.
        :return: Population with normalized fitness scores.
        """
        max_fitness = max([fit_score for _, fit_score in population])
        normalized_fitness = []
        for individual, fitness in population:
            normalized_fitness.append((individual, fitness/max_fitness))
        return normalized_fitness

    def _selection(self, population, one_one_tournament=False):
        """
        Tournament selection of individuals to reproduction.
        Firstly, a pair of individuals need to be chosen for a tournament.
        They are selected on the basis of their fitness score, meaning that individuals
        with better score are more likely to participate in the tournament, and thus
        be a part of next generation.
        The one (individual in the tournament) with better fitness score
        is appended for next population.
        :param population: Population of individuals
        :param one_one_tournament default=False: If True, it is possible
        for the same individual to be chosen to battle himself. This gives
        a chance for the worst individual to be a port of next population.
        :returns: New population.
        """

        individuals_w_fitness = self._fitnesses(population)  # calculate fitness function values for each individual

        normalized_fitness = self._normalize_fitness(individuals_w_fitness)  # normalize fitness

        tournament_results = []  # list containing population after tournaments
        while len(tournament_results) < self.population_size:  # conduct as many tournaments to fill population
            tournament_individuals = copy.copy(normalized_fitness)
            # choose 2 individuals randomly but with weights
            one = random.choices(tournament_individuals, weights=[pop[1] for pop in tournament_individuals], k=1)
            if not one_one_tournament:
                # remove chosen individual so he is not chosen again to battle himself
                tournament_individuals.remove(one[0])
            two = random.choices(tournament_individuals, weights=[pop[1] for pop in tournament_individuals], k=1)

            one_fit_score = one[0][1]
            two_fit_score = two[0][1]
            # choose individual with better fitness
            if one_fit_score >= two_fit_score:
                tournament_results.append(one[0][0])
            else:
                tournament_results.append(two[0][0])
        return tournament_results

    def _mutation(self, population, allow_invalid=False):
        """
        Perform mutation on individuals.
        :param population: Population of individuals
        :param allow_invalid: default=False: Bool value indicating whether the algorithm allows the mutated
        individual to be of invalid type, meaning that two vertices connected with an edge are of the same color.
        It might allow the population to leave local minimum in order to find the global minimum.
        :return: Return mutated population
        """
        population_after_mutation = []
        for individual in population:  # iterate through individuals
            prob_mutation = random.uniform(0, 1)  # probability of mutation
            if prob_mutation <= self.mutation_probability:
                for gene_position in range(0, len(individual)):  # iterate through genes
                    gene_mutation = random.uniform(0, 1)  # probability of mutation of specific gene
                    if gene_mutation <= self.gene_mutation_probability:  # if selected gene is to be mutated
                        available_colors = set(individual)  # set of available colors
                        # it contains only these colors currently used in individual
                        # mutate gen = change the color of the vertex.

                        if allow_invalid:  # allow invalid chromosomes (individuals) to among the mutated population
                            random_color = random.randint(0, len(self.graph.vertices))
                            new_individual = self._mutate_gene(individual, gene_position, random_color)
                            individual = new_individual
                        else:
                            invalid_color = True

                            while invalid_color:  # mutate gene till you get a valid individual
                                random_color = random.choice(tuple(available_colors))  # optimization purpose. Read [1]

                                new_individual = self._mutate_gene(individual, gene_position, random_color)

                                # if new individual is valid append him and break the loop
                                if self._is_valid(new_individual):
                                    invalid_color = False
                                    individual = new_individual  # replace individual with the mutated one
            population_after_mutation.append(individual)  # append individual

        return population_after_mutation

    def _mutate_gene(self, individual, gene_position, mutated_gene):
        """
        Mutates the gene of selected individual.
        :param individual: Individual with genes and fitness score
        :param gene_position: Position of the gene to mutate
        :param mutated_gene: Color of a vertex that is to be changed
        :return: Individual with mutated gene at position gene_position
        """
        individual[gene_position] = mutated_gene  # replace old gene with new one
        return individual

    def _best(self, population):
        """
        Choose the best individual from set of acceptable solutions and
        return him with his fitness. If no acceptable solution can be found
        among population (meaning that individuals mutated into bad),
        the appropriate message will be displayed.
        :param population: Population
        :returns: Best, acceptable individual from the population
        """
        try:
            acceptable_w_fitness = self._fitnesses(population)
            pop_fitness = sorted(acceptable_w_fitness, key=lambda t: t[1], reverse=True)  # sort by descending score

            best_individual = pop_fitness[0][0]  # first element and its individual without fitness score
            best_fitness = pop_fitness[0][1]  # get fitness score of best individual

            for population_and_fitness in pop_fitness:
                used_colors = []
                for color in population_and_fitness[0]:
                    if color not in used_colors:
                        used_colors.append(color)
                current_fitness_score = population_and_fitness[1]
                if current_fitness_score > best_fitness:  # if current population has better fitness score
                    best_individual = population_and_fitness[0]
                    best_fitness = current_fitness_score
        except IndexError:
            print('No best individual could be found.')
            best_individual, best_fitness = (0.0), 0.0

        return best_individual[:], best_fitness

    def _load_graph(self, graph_type: str, N:int = 25) -> None:
        """
        LOAD GRAPH.
        :param N: Number of vertices
        :return: None
        """
        # each vertex will randomly choose 2 for connection
        graph = Graph()

        # python 3.10 implements match case

        if graph_type == 'complete':
            graph._complete(N)
        elif graph_type == 'bipartite':
            graph._type_bipartite = True
            graph._bipartite(N)
        elif graph_type == 'random':
            graph._random(N)
        elif graph_type == 'random groups':
            graph._random_groups(N)
        else:
            graph._random_groups(N)

        return graph


def plot_graph(graph, best_edges, plot_name: str) -> None:
    """
    Method used to plot graph
    :param graph: Graph
    :param best_edges: Edges
    :param plot_name: Name of the plot
    :return: None
    """
    colors = dict(enumerate(palette_data_science_25))

    # create graph
    network = nx.Graph()

    # add nodes
    network.add_nodes_from(graph.vertices)

    # add edges
    if graph._type_bipartite:
        for edge in graph.edges:
            if edge[0] in graph.U:
                network.add_edge(edge[0], edge[1], color='#29526D')
            elif edge[0] in graph.V:
                network.add_edge(edge[0], edge[1], color='#AA3C39')
    else:
        for edge in graph.edges:
            network.add_edge(edge[0], edge[1])

    nx.set_node_attributes(network, None, "color")  # set attributes for all nodes | color is None

    for vertex, vert_color in enumerate(best_edges, 1):
        color = colors[vert_color-1]  # nodes are in range 1-25, where colors have indices in range 0-24
        network.nodes[vertex]['color'] = color

    colors_2 = [network.nodes[n]['color'] for n in network.nodes()]

    if graph._type_bipartite:
        colors = ['#29526D', '#AA3C39']
        pos = nx.bipartite_layout(graph.vertices, graph.U)
        nx.draw(network, pos, edge_color=colors, with_labels=True, node_color=colors_2, font_size=10, font_color="white")
    else:
        nx.draw_circular(network, with_labels=True, node_color=colors_2, font_size=10, font_color="white")

    n_components = plot_name.split('_')

    title = f"Type: {n_components[0]}\nPopulation size: {n_components[1]}\nIndividual | gene mutation probabilities: {n_components[2]} | {n_components[3]}"

    plt.title(title, fontsize=8.5, loc='left')

    plot_name = 'graph_' + plot_name

    plt.savefig(plot_name, dpi=150, bbox_inches='tight')

def save_results(best_individual, graph, exec_time, n_iterations, n_starting_colors) -> None:
    """
    Saves logs of an algorithm run to the .txt file
    :param best_individual: Best individual of algorithm run
    :param graph: Object of Graph with edges and vertices
    :param exec_time: Execution time of algorithm
    :param n_iterations: Number of iterations
    :param n_starting_colors: Number of colors used in generated
    graph at the beginning of the algorithm.
    :return: None
    """
    log_name = "logs_" + datetime.datetime.now().strftime("%b_%d_%Y-%H-%M-%S")
    with open(f'{log_name}.txt', 'w') as f_handle:
        logs = [f'Algorithm found the best individual in {n_iterations} iterations.\n']
        logs.append('Execution time {:0.2f} [s]\n'.format(exec_time))
        logs.append(f'At the beginning graph had: {n_starting_colors} unique colors\n')
        logs.append(f'Graph was minimized to only: {len(set(best_individual))} unique colors\n')
        logs.append(f'Best individual is best represented by the sequence: {best_individual}\n')
        logs.append(f'Vertices of generated graph: {graph.vertices}\n')
        logs.append(f'Edges of generated graph: {graph.edges}\n')
        for line in logs:
            f_handle.write(line)
        f_handle.close()

def plot_iterations_colors(results, N, plot_name):
    # SCORE MODEL #
    data = pd.DataFrame(columns=['Population'])

    for population, _ in results:
        number_of_unique_colors = len(set(population))
        data = data.append({'Unique colors': number_of_unique_colors}, ignore_index=True)

    ax = plt.figure(figsize=(8, 6)).add_subplot()

    ax.scatter(x=range(0, len(data)), y=data['Unique colors'], s=9, c='#ffa600')
    ax.set_yticks(np.arange(0, N+1, 1))
    ax.set_xticks(np.arange(0, len(results), 100))

    plt.xlabel('Number of iterations', fontsize=12)
    plt.ylabel('Number of colors', fontsize=12)

    plt.grid(color='grey', linestyle='-', linewidth=0.25)

    plt.box(False)  # borderless

    n_components = plot_name.split('_')

    title = f"Type: {n_components[0]}\nPopulation size: {n_components[1]}\nIndividual | gene mutation probabilities: {n_components[2]} | {n_components[3]}"

    plt.title(title, fontsize=8.5, loc='left')
    plot_name = 'iterations_' + plot_name

    plt.savefig(plot_name, dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    N_of_vertices = 25
    max_iterations = 900
    population_size = 100
    mutation_probability = 0.45
    gene_mutation_probability = 0.12
    max_no_improvements = 900  # maximum number of iterations in which the algorithm did not find better solution
    graph_type = 'bipartite'

    colorizer = GraphColorizer(max_iterations,
                               population_size,
                               mutation_probability,
                               gene_mutation_probability,
                               graph_type,
                               max_no_improvements)

    start_time = time.time()  # start time execution measurement

    results, best_fitness, best_individual, n_iterations, n_starting_colors = colorizer.run_algorithm()

    exec_time = time.time() - start_time  # end time execution measurement

    print("\nExecution time: {:.4f} [s]".format(exec_time))
    print(f'Number of iterations: {n_iterations}\n')

    print(f'Best individual: {best_individual}')
    print(f'Colors used: {len(set(best_individual))}')

    # Plot results #

    plot_name = f'{graph_type}_{population_size}_{mutation_probability}_{gene_mutation_probability}_{max_iterations}.png'

    plot_graph(colorizer.graph, best_individual, plot_name)

    plot_iterations_colors(results,
                N_of_vertices,
                plot_name)

    # Save report #

    save_results(best_individual,
                 colorizer.graph,
                 exec_time,
                 n_iterations,
                 n_starting_colors)

"""
Resources and annotations
[1] https://stackoverflow.com/questions/15837729/random-choice-from-set-python
[2] https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3
"""
