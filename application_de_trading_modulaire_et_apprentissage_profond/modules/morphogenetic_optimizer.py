import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Classe MorphogeneticOptimizer
class MorphogeneticOptimizer(nn.Module):
    """Inspiré du développement biologique des organismes avec auto-adaptation"""
    def __init__(self, initial_genome, mutation_rate=0.01, min_mutation_rate=0.001, max_mutation_rate=0.1, adaptation_rate=0.05):
        super().__init__()
        self.genome = initial_genome
        self.mutation_rate = mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        self.generation = 0

    def develop_network(self):
        layers = []
        for gene in self.genome:
            if gene[0] > 0.5:  # Add layer
                layers.append(nn.Linear(int(gene[1]*100), int(gene[2]*100)))
                if gene[3] > 0.7:  # Add activation
                    layers.append(nn.ReLU() if gene[4] > 0.5 else nn.Tanh())
                if gene[5] > 0.5:  # Add normalization
                    layers.append(nn.BatchNorm1d(int(gene[2]*100)))
                if gene[6] > 0.5:  # Add dropout
                    layers.append(nn.Dropout(p=gene[7]))
        return nn.Sequential(*layers)

    def mutate(self):
        """Mutate the genome to explore new architectures with adaptive mutation rate."""
        for gene in self.genome:
            if np.random.rand() < self.mutation_rate:
                gene[np.random.randint(0, len(gene))] = np.random.rand()
        self.generation += 1

    def adapt_mutation_rate(self, current_performance):
        """Adapts mutation rate based on performance trends."""
        self.performance_history.append(current_performance)
        
        # Besoin d'au moins 2 points de données pour adapter
        if len(self.performance_history) < 2:
            return
        
        # Calculer la tendance de performance
        performance_delta = self.performance_history[-1] - self.performance_history[-2]
        
        # Si la performance s'améliore, réduire le taux de mutation pour exploiter
        if performance_delta > 0:
            self.mutation_rate = max(self.min_mutation_rate, 
                                    self.mutation_rate * (1 - self.adaptation_rate))
        # Si la performance se dégrade, augmenter le taux de mutation pour explorer
        else:
            self.mutation_rate = min(self.max_mutation_rate, 
                                    self.mutation_rate * (1 + self.adaptation_rate))
    
    def crossover(self, partner):
        """Combine two genomes to create a new one."""
        new_genome = []
        for g1, g2 in zip(self.genome, partner.genome):
            # Crossover avec variation aléatoire pour plus de diversité
            alpha = np.random.rand()  # Facteur de mélange aléatoire
            new_gene = [alpha * a + (1 - alpha) * b for a, b in zip(g1, g2)]
            
            # Ajouter une petite mutation pour éviter la convergence prématurée
            if np.random.rand() < self.mutation_rate / 2:
                idx = np.random.randint(0, len(new_gene))
                new_gene[idx] = np.random.rand()
                
            new_genome.append(new_gene)
        
        # Hériter du taux de mutation du parent le plus performant
        if len(self.performance_history) > 0 and len(partner.performance_history) > 0:
            if self.performance_history[-1] > partner.performance_history[-1]:
                mutation_rate = self.mutation_rate
            else:
                mutation_rate = partner.mutation_rate
        else:
            mutation_rate = (self.mutation_rate + partner.mutation_rate) / 2
            
        child = MorphogeneticOptimizer(new_genome, mutation_rate, 
                                      self.min_mutation_rate, self.max_mutation_rate, 
                                      self.adaptation_rate)
        return child
        
    def evaluate_fitness(self, data, target, epochs=5):
        """Évalue la fitness du réseau développé sur des données."""
        network = self.develop_network()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Conversion des données en tenseurs
        data_tensor = torch.FloatTensor(data)
        target_tensor = torch.FloatTensor(target)
        
        # Entraînement rapide
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = network(data_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Retourner l'inverse de la perte moyenne comme fitness
        avg_loss = sum(losses) / len(losses)
        fitness = 1.0 / (avg_loss + 1e-10)  # Éviter division par zéro
        
        # Mettre à jour l'historique des performances
        self.performance_history.append(fitness)
        
        # Adapter le taux de mutation
        if len(self.performance_history) >= 2:
            self.adapt_mutation_rate(fitness)
            
        return fitness

# Classe ConsciousLayer
class ConsciousLayer(nn.Module):
    """Mécanisme d'auto-réflexion du modèle"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.self_query = nn.Linear(dim, dim)
        self.memory_proj = nn.Linear(dim * 2, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, memory):
        batch_size = x.size(0)
        q = self.self_query(x).view(batch_size, self.num_heads, self.head_dim, -1)
        k = x.view(batch_size, self.num_heads, self.head_dim, -1)
        v = x.view(batch_size, self.num_heads, self.head_dim, -1)

        attn = F.softmax(torch.einsum('bhid,bhjd->bhij', q, k), dim=-1)
        context = torch.einsum('bhij,bhjd->bhid', attn, v).contiguous().view(batch_size, self.dim, -1)

        combined = torch.cat([context, memory], dim=-1)
        output = self.memory_proj(combined)

        output = self.layer_norm(output)
        output = self.activation(output)
        output = self.dropout(output)

        return output

# Classe NTMController
class NTMController(nn.Module):
    """Mémoire externe différentiable avec mécanismes d'accès et oubli sélectif"""
    def __init__(self, mem_size=256, mem_dim=32, num_read_heads=1, num_write_heads=1, forget_factor=0.1):
        super().__init__()
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.forget_factor = forget_factor

        # Mémoire principale
        self.memory = nn.Parameter(torch.randn(mem_size, mem_dim))
        
        # Compteurs d'utilisation pour l'oubli sélectif
        self.register_buffer('usage_counter', torch.zeros(mem_size))
        self.register_buffer('age_factor', torch.zeros(mem_size))
        
        # Têtes de lecture et d'écriture
        self.read_heads = nn.ModuleList([nn.Linear(mem_dim, mem_dim) for _ in range(num_read_heads)])
        self.write_heads = nn.ModuleList([nn.Linear(mem_dim, mem_dim) for _ in range(num_write_heads)])
        
        # Mécanisme d'oubli sélectif
        self.forget_gate = nn.Sequential(
            nn.Linear(mem_dim, 1),
            nn.Sigmoid()
        )
        
        # Mécanisme d'allocation
        self.allocation_gate = nn.Sequential(
            nn.Linear(mem_dim, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            init.xavier_uniform_(self.memory)
            for layer in self.read_heads + self.write_heads:
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0)

    def address_memory(self, key, strength=10):
        # Calcul de similarité cosinus
        similarity = F.cosine_similarity(self.memory, key.unsqueeze(1), dim=-1)
        
        # Ajustement par l'âge et l'utilisation
        adjusted_similarity = similarity - self.age_factor * self.forget_factor
        
        # Normalisation avec softmax
        weights = F.softmax(strength * adjusted_similarity, dim=-1)
        return weights

    def read_from_memory(self, weights):
        # Mise à jour des compteurs d'utilisation pour les emplacements lus
        self.usage_counter += weights
        return torch.matmul(weights, self.memory)

    def write_to_memory(self, weights, values):
        batch_size = values.size(0)
        
        # Calculer les vecteurs d'écriture pour chaque tête
        write_vectors = [head(values[:, i]) for i, head in enumerate(self.write_heads)]
        
        # Calculer le facteur d'oubli pour chaque emplacement de mémoire
        forget_weights = torch.zeros_like(weights)
        for i, vec in enumerate(write_vectors):
            # Déterminer quels emplacements de mémoire doivent être oubliés
            forget_strength = self.forget_gate(vec).view(-1)  # [batch_size]
            
            # Appliquer le facteur d'oubli aux poids
            forget_weights += weights * forget_strength.unsqueeze(1)  # [batch_size, mem_size]
        
        # Normaliser les poids d'oubli
        forget_weights = torch.clamp(forget_weights, 0, 1)
        
        # Appliquer l'oubli sélectif à la mémoire
        memory_retention = 1 - forget_weights.mean(dim=0).unsqueeze(1)  # [mem_size, 1]
        self.memory.data = self.memory.data * memory_retention
        
        # Écrire les nouvelles valeurs
        for i, vec in enumerate(write_vectors):
            write_strength = self.allocation_gate(vec).view(-1)  # [batch_size]
            self.memory.data = self.memory.data + torch.matmul(weights.t() * write_strength.unsqueeze(0), vec.unsqueeze(1))
        
        # Mettre à jour les facteurs d'âge
        self.age_factor += 0.01  # Incrémenter l'âge de tous les emplacements
        self.age_factor *= (1 - weights.mean(dim=0))  # Réinitialiser l'âge des emplacements récemment écrits

    def forget_unused_memory(self, threshold=5.0):
        """Oublie les emplacements de mémoire peu utilisés."""
        # Identifier les emplacements peu utilisés
        unused_locations = (self.usage_counter < threshold).float().unsqueeze(1)
        
        # Réinitialiser ces emplacements avec un bruit aléatoire
        random_values = torch.randn_like(self.memory)
        self.memory.data = self.memory.data * (1 - unused_locations) + random_values * unused_locations
        
        # Réinitialiser les compteurs d'utilisation pour ces emplacements
        self.usage_counter *= (1 - unused_locations.squeeze(1))
        self.age_factor *= (1 - unused_locations.squeeze(1))

    def forward(self, keys, values):
        # Calculer les poids de lecture pour chaque clé
        read_weights = [self.address_memory(key) for key in keys]
        
        # Lire depuis la mémoire
        read_vectors = [self.read_from_memory(rw) for rw in read_weights]
        
        # Écrire dans la mémoire
        self.write_to_memory(read_weights[0], values)
        
        # Périodiquement oublier la mémoire non utilisée
        if torch.rand(1).item() < 0.05:  # 5% de chance à chaque forward pass
            self.forget_unused_memory()
        
        return read_vectors

# Classe LIFNeuron
class LIFNeuron(nn.Module):
    """Neurone à intégrateur et déclencheur (Leaky Integrate-and-Fire)"""
    def __init__(self, threshold=1.0, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.register_buffer('membrane', torch.zeros(1))

    def forward(self, inputs):
        if inputs.dim() > 1:
            batch_size = inputs.size(0)
            self.membrane = self.membrane.repeat(batch_size)
            inputs_sum = inputs.sum(dim=1)
        else:
            inputs_sum = inputs.sum()

        self.membrane = self.decay * self.membrane + inputs_sum
        spike = (self.membrane >= self.threshold).float()
        self.membrane -= spike * self.threshold

        return spike

    def reset(self):
        """Reset the membrane potential."""
        self.membrane.zero_()

# Classe SignalFusion
class SignalFusion(nn.Module):
    """Combine RSI, MACD, Bollinger via attention mechanism"""
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Projections pour chaque type d'indicateur
        self.rsi_proj = nn.Linear(input_dim, hidden_dim)
        self.macd_proj = nn.Linear(input_dim, hidden_dim)
        self.bollinger_proj = nn.Linear(input_dim, hidden_dim)
        
        # Mécanisme d'attention multi-têtes
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        
        # Couches de sortie
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, rsi, macd, bollinger):
        # Projections des indicateurs
        rsi_emb = self.rsi_proj(rsi)
        macd_emb = self.macd_proj(macd)
        bollinger_emb = self.bollinger_proj(bollinger)
        
        # Concaténation des embeddings pour l'attention
        # [seq_len, batch, hidden_dim]
        indicators = torch.stack([rsi_emb, macd_emb, bollinger_emb], dim=0)
        
        # Mécanisme d'attention
        attn_output, _ = self.attention(indicators, indicators, indicators)
        attn_output = self.dropout(attn_output)
        out1 = self.norm1(indicators + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output)
        out2 = self.norm2(out1 + ffn_output)
        
        # Moyenne sur la dimension des indicateurs
        fused = torch.mean(out2, dim=0)
        
        # Couche de sortie
        output = self.output_layer(fused)
        
        return output