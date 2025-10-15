"""
Choquet Classifier Implementation
Replication of Figure 3(a) from "Nonlinear Classification by Genetic Algorithm with Signed Fuzzy Measure"

This script implements a Genetic Algorithm to find the optimal Choquet hyperplane
for binary classification in 2D space.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict
import json

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class ChoquetClassifier:
    """
    Implements the Choquet integral-based classifier with Genetic Algorithm optimization.
    """
    
    def __init__(self, population_size: int = 100, penalty_c: int = 111, 
                 bits_per_param: int = 10, mutation_rate: float = 0.01):
        """
        Initialize the Choquet classifier.
        
        Args:
            population_size: Size of the genetic algorithm population
            penalty_c: Penalty factor for misclassified points
            bits_per_param: Number of bits per parameter in chromosome encoding
            mutation_rate: Probability of mutation per bit
        """
        self.population_size = population_size
        self.penalty_c = penalty_c
        self.bits_per_param = bits_per_param
        self.mutation_rate = mutation_rate
        self.chromosome_length = 8 * bits_per_param  # 8 parameters total
        
        # Parameters from paper's Table III (final results)
        self.paper_params = {
            'mu_1': 0.1802,
            'mu_2': 0.5460, 
            'mu_12': 0.1389,
            'B': 0.0917,
            'a': [0.0, 0.0],
            'b': [1.0, 1.0]
        }
        
    def generate_data(self, n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic 2D classification data using the paper's final parameters.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (data_points, labels) where labels are 1 for Class A, -1 for Class A'
        """
        print("Generating synthetic data using paper's final parameters...")
        
        # Generate random points in [0, 1] x [0, 1]
        data = np.random.uniform(0, 1, (n_samples, 2))
        
        # Use paper's final parameters to determine class labels
        mu_1, mu_2, mu_12 = self.paper_params['mu_1'], self.paper_params['mu_2'], self.paper_params['mu_12']
        B = self.paper_params['B']
        a = np.array(self.paper_params['a'])
        b = np.array(self.paper_params['b'])
        
        labels = []
        for point in data:
            choquet_value = self.choquet_integral_2d(point, a, b, [mu_1, mu_2, mu_12])
            if choquet_value >= B:
                labels.append(1)  # Class A
            else:
                labels.append(-1)  # Class A'
        
        labels = np.array(labels)
        
        # Verify we get approximately the right distribution (155 Class A, 45 Class A')
        class_a_count = np.sum(labels == 1)
        class_a_prime_count = np.sum(labels == -1)
        
        print(f"Generated {n_samples} samples:")
        print(f"  Class A: {class_a_count} samples")
        print(f"  Class A': {class_a_prime_count} samples")
        
        return data, labels
    
    def choquet_integral_2d(self, x: np.ndarray, a: np.ndarray, b: np.ndarray, 
                           mu: List[float]) -> float:
        """
        Calculate the Choquet integral for a 2D data point.
        
        Args:
            x: 2D data point [x1, x2]
            a: Vector a = [a1, a2]
            b: Vector b = [b1, b2] 
            mu: Measure values [mu_1, mu_2, mu_12]
            
        Returns:
            Choquet integral value
        """
        mu_1, mu_2, mu_12 = mu
        
        # Compute integrand: h_i = a_i + b_i * x_i
        h = a + b * x
        h1, h2 = h[0], h[1]
        
        # Sort h values and apply Choquet integral formula
        if h1 <= h2:
            # h1 <= h2: C = h1 * mu_12 + (h2 - h1) * mu_2
            choquet_value = h1 * mu_12 + (h2 - h1) * mu_2
        else:
            # h2 < h1: C = h2 * mu_12 + (h1 - h2) * mu_1  
            choquet_value = h2 * mu_12 + (h1 - h2) * mu_1
            
        return choquet_value
    
    def decode_chromosome(self, chromosome: str) -> Dict[str, float]:
        """
        Decode binary chromosome to real-valued parameters.
        
        Args:
            chromosome: Binary string representing the chromosome
            
        Returns:
            Dictionary of decoded parameters
        """
        params = {}
        
        # Decode each parameter (8 parameters total)
        for i in range(8):
            start_idx = i * self.bits_per_param
            end_idx = start_idx + self.bits_per_param
            
            # Extract binary segment and convert to decimal
            binary_segment = chromosome[start_idx:end_idx]
            decimal_value = int(binary_segment, 2) / (2**self.bits_per_param - 1)
            
            if i in [0, 1, 2]:  # mu_1, mu_2, mu_12 - fuzzy measures in [0, 1]
                param_value = decimal_value
            elif i == 3:  # B - threshold in [0, 1]
                param_value = decimal_value
            else:  # a0, a1, b0, b1 - in [-1, 1]
                param_value = 2 * (decimal_value - 0.5)
            
            if i == 0:
                params['mu_1'] = param_value
            elif i == 1:
                params['mu_2'] = param_value
            elif i == 2:
                params['mu_12'] = param_value
            elif i == 3:
                params['B'] = param_value
            elif i == 4:
                params['a0'] = param_value
            elif i == 5:
                params['a1'] = param_value
            elif i == 6:
                params['b0'] = param_value
            elif i == 7:
                params['b1'] = param_value
                
        return params
    
    def fitness_function(self, chromosome: str, data: np.ndarray, 
                        labels: np.ndarray) -> float:
        """
        Calculate fitness (penalized total signed Choquet distance D_c).
        
        Args:
            chromosome: Binary chromosome string
            data: Training data points
            labels: Class labels (1 for Class A, -1 for Class A')
            
        Returns:
            Fitness score (higher is better)
        """
        # Decode chromosome to get parameters
        params = self.decode_chromosome(chromosome)
        
        # Extract parameters
        mu_1, mu_2, mu_12 = params['mu_1'], params['mu_2'], params['mu_12']
        B = params['B']
        a = np.array([params['a0'], params['a1']])
        b = np.array([params['b0'], params['b1']])
        
        # Ensure measure values are in valid range [0, 1] and monotonic
        mu_1 = max(0, min(1, mu_1))
        mu_2 = max(0, min(1, mu_2))
        mu_12 = max(0, min(1, mu_12))
        
        # Ensure monotonicity: mu_12 >= max(mu_1, mu_2)
        mu_12 = max(mu_12, max(mu_1, mu_2))
        
        total_distance = 0.0
        correct_classifications = 0
        
        for i, point in enumerate(data):
            # Calculate Choquet integral for this point
            choquet_value = self.choquet_integral_2d(point, a, b, [mu_1, mu_2, mu_12])
            
            # Calculate signed distance from decision boundary
            signed_distance = choquet_value - B
            
            # Apply penalty for misclassification
            if labels[i] == 1:  # Class A
                if choquet_value < B:  # Misclassified
                    signed_distance *= self.penalty_c
                else:
                    correct_classifications += 1
            else:  # Class A'
                if choquet_value >= B:  # Misclassified
                    signed_distance *= self.penalty_c
                else:
                    correct_classifications += 1
                    
            total_distance += signed_distance
            
        # Add bonus for correct classifications
        accuracy_bonus = correct_classifications * 1000
        
        return total_distance + accuracy_bonus
    
    def create_random_chromosome(self) -> str:
        """Create a random binary chromosome."""
        return ''.join(random.choice('01') for _ in range(self.chromosome_length))
    
    def roulette_wheel_selection(self, population: List[str], fitness_scores: List[float]) -> str:
        """
        Select parent using roulette wheel selection.
        
        Args:
            population: List of chromosomes
            fitness_scores: Corresponding fitness scores
            
        Returns:
            Selected chromosome
        """
        # Convert fitness to selection probabilities
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choice(population)
            
        probabilities = [f / total_fitness for f in fitness_scores]
        
        # Roulette wheel selection
        r = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return population[i]
                
        return population[-1]  # Fallback
    
    def crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """
        Perform single-point crossover between two parents.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two offspring chromosomes
        """
        # Random crossover point
        crossover_point = random.randint(1, self.chromosome_length - 1)
        
        # Create offspring
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return offspring1, offspring2
    
    def mutate(self, chromosome: str) -> str:
        """
        Apply mutation to a chromosome.
        
        Args:
            chromosome: Input chromosome
            
        Returns:
            Mutated chromosome
        """
        chromosome_list = list(chromosome)
        
        for i in range(len(chromosome_list)):
            if random.random() < self.mutation_rate:
                chromosome_list[i] = '1' if chromosome_list[i] == '0' else '0'
                
        return ''.join(chromosome_list)
    
    def run_genetic_algorithm(self, data: np.ndarray, labels: np.ndarray, 
                            max_generations: int = 100) -> Dict:
        """
        Run the genetic algorithm to find optimal parameters.
        
        Args:
            data: Training data
            labels: Class labels
            max_generations: Maximum number of generations
            
        Returns:
            Dictionary with best parameters and evolution history
        """
        print(f"Starting Genetic Algorithm with population size {self.population_size}")
        print(f"Chromosome length: {self.chromosome_length} bits")
        
        # Initialize population
        population = [self.create_random_chromosome() for _ in range(self.population_size)]
        
        best_fitness_history = []
        best_params_history = []
        no_improvement_count = 0
        
        for generation in range(max_generations):
            # Evaluate fitness for all chromosomes
            fitness_scores = []
            for chromosome in population:
                fitness = self.fitness_function(chromosome, data, labels)
                fitness_scores.append(fitness)
            
            # Find best chromosome
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_chromosome = population[best_idx]
            best_params = self.decode_chromosome(best_chromosome)
            
            best_fitness_history.append(best_fitness)
            best_params_history.append(best_params.copy())
            
            print(f"Generation {generation + 1}: Best fitness = {best_fitness:.6f}")
            
            # Check for improvement
            if generation > 0 and best_fitness <= best_fitness_history[-2]:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
                
            # Stopping condition: no improvement for 10 generations
            if no_improvement_count >= 10:
                print(f"Stopping early: no improvement for {no_improvement_count} generations")
                break
            
            # Create new population
            new_population = []
            
            # Keep best half of current population
            sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
            elite_size = self.population_size // 2
            for i in range(elite_size):
                new_population.append(population[sorted_indices[i]])
            
            # Generate offspring to fill remaining slots
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.roulette_wheel_selection(population, fitness_scores)
                parent2 = self.roulette_wheel_selection(population, fitness_scores)
                
                # Create offspring
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                # Apply mutation
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            # Ensure population size is correct
            population = new_population[:self.population_size]
        
        print(f"Genetic Algorithm completed after {generation + 1} generations")
        print(f"Final best fitness: {best_fitness:.6f}")
        
        return {
            'best_params': best_params,
            'best_fitness': best_fitness,
            'fitness_history': best_fitness_history,
            'params_history': best_params_history,
            'generations': generation + 1
        }
    
    def plot_results(self, data: np.ndarray, labels: np.ndarray, 
                    best_params: Dict, save_path: str = 'figure_3a_replication.png'):
        """
        Plot the classification results and decision boundary.
        
        Args:
            data: Training data points
            labels: Class labels
            best_params: Best parameters from GA
            save_path: Path to save the plot
        """
        print("Generating plot...")
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot data points
        class_a_mask = labels == 1
        class_a_prime_mask = labels == -1
        
        plt.scatter(data[class_a_mask, 0], data[class_a_mask, 1], 
                   c='red', marker='o', label='Class A', s=50, alpha=0.7)
        plt.scatter(data[class_a_prime_mask, 0], data[class_a_prime_mask, 1], 
                   c='blue', marker='s', label="Class A'", s=50, alpha=0.7)
        
        # Create grid for decision boundary
        x_min, x_max = 0, 1
        y_min, y_max = 0, 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Calculate decision boundary
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        decision_values = []
        
        mu_1, mu_2, mu_12 = best_params['mu_1'], best_params['mu_2'], best_params['mu_12']
        B = best_params['B']
        a = np.array([best_params['a0'], best_params['a1']])
        b = np.array([best_params['b0'], best_params['b1']])
        
        for point in grid_points:
            choquet_value = self.choquet_integral_2d(point, a, b, [mu_1, mu_2, mu_12])
            decision_values.append(choquet_value - B)
        
        decision_values = np.array(decision_values).reshape(xx.shape)
        
        # Plot decision boundary
        plt.contour(xx, yy, decision_values, levels=[0], colors='black', 
                   linewidths=2, linestyles='--', label='Choquet Hyperplane')
        
        # Formatting
        plt.xlabel('Feature x1', fontsize=12)
        plt.ylabel('Feature x2', fontsize=12)
        plt.title('Replication of Figure 3(a): Choquet Classifier Results', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved as {save_path}")
    
    def calculate_accuracy(self, data: np.ndarray, labels: np.ndarray, 
                          params: Dict) -> float:
        """
        Calculate classification accuracy.
        
        Args:
            data: Test data
            labels: True labels
            params: Classifier parameters
            
        Returns:
            Classification accuracy
        """
        correct = 0
        total = len(data)
        
        mu_1, mu_2, mu_12 = params['mu_1'], params['mu_2'], params['mu_12']
        B = params['B']
        a = np.array([params['a0'], params['a1']])
        b = np.array([params['b0'], params['b1']])
        
        for i, point in enumerate(data):
            choquet_value = self.choquet_integral_2d(point, a, b, [mu_1, mu_2, mu_12])
            
            predicted_label = 1 if choquet_value >= B else -1
            if predicted_label == labels[i]:
                correct += 1
                
        return correct / total


def main():
    """Main function to run the complete experiment."""
    print("=" * 60)
    print("Choquet Classifier - Replication of Figure 3(a)")
    print("=" * 60)
    
    # Initialize classifier
    classifier = ChoquetClassifier()
    
    # Generate synthetic data
    data, labels = classifier.generate_data(n_samples=200)
    
    # Run genetic algorithm
    results = classifier.run_genetic_algorithm(data, labels, max_generations=100)
    
    # Print results
    print("\n" + "=" * 40)
    print("FINAL RESULTS")
    print("=" * 40)
    
    best_params = results['best_params']
    print("Best parameters found by GA:")
    for param, value in best_params.items():
        print(f"  {param}: {value:.6f}")
    
    print(f"\nBest fitness: {results['best_fitness']:.6f}")
    print(f"Generations: {results['generations']}")
    
    # Calculate accuracy
    accuracy = classifier.calculate_accuracy(data, labels, best_params)
    print(f"Training accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Compare with paper's parameters
    print("\n" + "=" * 40)
    print("COMPARISON WITH PAPER'S PARAMETERS")
    print("=" * 40)
    print("Parameter    | Paper Value | GA Result  | Difference")
    print("-" * 50)
    
    paper_mu_1, paper_mu_2, paper_mu_12 = classifier.paper_params['mu_1'], classifier.paper_params['mu_2'], classifier.paper_params['mu_12']
    paper_B = classifier.paper_params['B']
    
    print(f"mu_1        | {paper_mu_1:10.4f} | {best_params['mu_1']:10.4f} | {abs(paper_mu_1 - best_params['mu_1']):10.4f}")
    print(f"mu_2        | {paper_mu_2:10.4f} | {best_params['mu_2']:10.4f} | {abs(paper_mu_2 - best_params['mu_2']):10.4f}")
    print(f"mu_12       | {paper_mu_12:10.4f} | {best_params['mu_12']:10.4f} | {abs(paper_mu_12 - best_params['mu_12']):10.4f}")
    print(f"B           | {paper_B:10.4f} | {best_params['B']:10.4f} | {abs(paper_B - best_params['B']):10.4f}")
    
    # Plot results
    classifier.plot_results(data, labels, best_params)
    
    # Save results to file
    results_summary = {
        'best_params': best_params,
        'best_fitness': results['best_fitness'],
        'accuracy': accuracy,
        'generations': results['generations'],
        'paper_params': classifier.paper_params
    }
    
    with open('results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to results_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
