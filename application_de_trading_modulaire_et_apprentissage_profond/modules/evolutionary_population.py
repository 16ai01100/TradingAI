import numpy as np
import torch
import torch.nn as nn
import logging
import os
from collections import deque
import random
from .morphogenetic_optimizer import MorphogeneticOptimizer

class EvolutionaryPopulation:
    """
    Gère une population d'optimiseurs morphogénétiques et applique des algorithmes
    évolutionnaires pour améliorer les performances du système de trading.
    """
    def __init__(self, population_size=20, genome_template=None, genome_size=8, gene_length=8, 
                 elite_size=2, tournament_size=3, crossover_prob=0.7, mutation_prob=0.1):
        """
        Initialise une population d'optimiseurs morphogénétiques.
        
        Args:
            population_size (int): Taille de la population
            genome_template (list): Modèle de génome à utiliser, si None un génome aléatoire est créé
            genome_size (int): Nombre de gènes dans chaque génome
            gene_length (int): Longueur de chaque gène
            elite_size (int): Nombre d'individus élites à préserver
            tournament_size (int): Taille du tournoi pour la sélection
            crossover_prob (float): Probabilité de croisement
            mutation_prob (float): Probabilité de mutation
        """
        self.population_size = population_size
        self.genome_size = genome_size
        self.gene_length = gene_length
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Initialisation de la population
        self.population = []
        self.fitness_scores = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.generation = 0
        self.fitness_history = deque(maxlen=100)
        
        # Créer la population initiale
        if genome_template is None:
            for _ in range(population_size):
                genome = [np.random.rand(gene_length) for _ in range(genome_size)]
                self.population.append(MorphogeneticOptimizer(genome))
        else:
            # Créer des variations du génome template
            for _ in range(population_size):
                genome = []
                for gene in genome_template:
                    # Copier le gène avec de petites variations
                    new_gene = gene.copy()
                    for i in range(len(new_gene)):
                        if np.random.rand() < 0.3:  # 30% de chance de mutation initiale
                            new_gene[i] = np.random.rand()
                    genome.append(new_gene)
                self.population.append(MorphogeneticOptimizer(genome))
        
        # Configuration du logging
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'evolutionary_population.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EvolutionaryPopulation')
        self.logger.info(f"Population évolutionnaire initialisée avec {population_size} individus")
    
    def evaluate_population(self, data, target):
        """
        Évalue la fitness de chaque individu dans la population.
        
        Args:
            data: Données d'entrée pour l'évaluation
            target: Cibles pour l'évaluation
            
        Returns:
            tuple: (meilleur_individu, meilleure_fitness)
        """
        self.fitness_scores = []
        
        for i, individual in enumerate(self.population):
            fitness = individual.evaluate_fitness(data, target)
            self.fitness_scores.append(fitness)
            
            # Mettre à jour le meilleur individu
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual
                self.logger.info(f"Nouvel individu optimal trouvé: fitness={fitness:.4f}")
        
        # Enregistrer l'historique de fitness
        avg_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
        self.fitness_history.append(avg_fitness)
        
        self.logger.info(f"Génération {self.generation}: fitness moyenne={avg_fitness:.4f}, meilleure={self.best_fitness:.4f}")
        
        return self.best_individual, self.best_fitness
    
    def selection_tournament(self):
        """
        Sélectionne un individu par tournoi.
        
        Returns:
            MorphogeneticOptimizer: Individu sélectionné
        """
        # Sélectionner aléatoirement des individus pour le tournoi
        tournament_indices = random.sample(range(self.population_size), self.tournament_size)
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        
        # Sélectionner le meilleur du tournoi
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        
        return self.population[winner_idx]
    
    def evolve(self):
        """
        Fait évoluer la population vers la génération suivante.
        """
        # Trier la population par fitness
        sorted_indices = sorted(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i], reverse=True)
        sorted_population = [self.population[i] for i in sorted_indices]
        
        # Conserver les élites
        new_population = sorted_population[:self.elite_size]
        
        # Créer le reste de la population par sélection, croisement et mutation
        while len(new_population) < self.population_size:
            # Sélection par tournoi
            parent1 = self.selection_tournament()
            parent2 = self.selection_tournament()
            
            # Croisement
            if random.random() < self.crossover_prob:
                child = parent1.crossover(parent2)
            else:
                # Pas de croisement, copier un parent
                child = parent1 if random.random() < 0.5 else parent2
            
            # Mutation
            if random.random() < self.mutation_prob:
                child.mutate()
            
            new_population.append(child)
        
        # Mettre à jour la population
        self.population = new_population
        self.generation += 1
        
        self.logger.info(f"Population évoluée vers la génération {self.generation}")
    
    def get_best_network(self):
        """
        Retourne le meilleur réseau développé.
        
        Returns:
            nn.Sequential: Meilleur réseau
        """
        if self.best_individual is None:
            return None
        
        return self.best_individual.develop_network()
    
    def get_diversity(self):
        """
        Calcule la diversité génétique de la population.
        
        Returns:
            float: Mesure de diversité
        """
        if not self.population:
            return 0
        
        # Calculer la distance moyenne entre les génomes
        total_distance = 0
        count = 0
        
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):
                # Calculer la distance entre les génomes i et j
                genome_i = self.population[i].genome
                genome_j = self.population[j].genome
                
                distance = 0
                for gene_i, gene_j in zip(genome_i, genome_j):
                    # Distance euclidienne entre les gènes
                    gene_distance = np.sqrt(np.sum((np.array(gene_i) - np.array(gene_j))**2))
                    distance += gene_distance
                
                total_distance += distance
                count += 1
        
        if count == 0:
            return 0
        
        return total_distance / count
    
    def inject_diversity(self, ratio=0.1):
        """
        Injecte de la diversité dans la population en remplaçant certains individus.
        
        Args:
            ratio (float): Proportion de la population à remplacer
        """
        num_to_replace = max(1, int(self.population_size * ratio))
        
        # Préserver les élites
        sorted_indices = sorted(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i], reverse=True)
        elite_indices = set(sorted_indices[:self.elite_size])
        
        # Sélectionner aléatoirement des individus à remplacer (sauf les élites)
        replaceable_indices = [i for i in range(self.population_size) if i not in elite_indices]
        replace_indices = random.sample(replaceable_indices, min(num_to_replace, len(replaceable_indices)))
        
        # Remplacer par de nouveaux individus aléatoires
        for idx in replace_indices:
            genome = [np.random.rand(self.gene_length) for _ in range(self.genome_size)]
            self.population[idx] = MorphogeneticOptimizer(genome)
        
        self.logger.info(f"Diversité injectée: {num_to_replace} individus remplacés")
    
    def adapt_parameters(self):
        """
        Adapte les paramètres évolutionnaires en fonction des tendances de fitness.
        """
        if len(self.fitness_history) < 10:
            return
        
        # Calculer la tendance de fitness sur les 10 dernières générations
        recent_fitness = list(self.fitness_history)[-10:]
        fitness_trend = (recent_fitness[-1] - recent_fitness[0]) / recent_fitness[0] if recent_fitness[0] != 0 else 0
        
        # Calculer la diversité actuelle
        diversity = self.get_diversity()
        
        # Adapter les paramètres en fonction de la tendance et de la diversité
        if fitness_trend < 0.01:  # Stagnation
            # Augmenter la mutation et le croisement pour explorer davantage
            self.mutation_prob = min(0.3, self.mutation_prob * 1.2)
            self.crossover_prob = min(0.9, self.crossover_prob * 1.1)
            
            # Si la diversité est faible, injecter de la diversité
            if diversity < 0.5:
                self.inject_diversity(0.2)
        else:  # Amélioration
            # Réduire progressivement la mutation pour exploiter
            self.mutation_prob = max(0.05, self.mutation_prob * 0.95)
            
        self.logger.info(f"Paramètres adaptés: mutation_prob={self.mutation_prob:.3f}, crossover_prob={self.crossover_prob:.3f}")