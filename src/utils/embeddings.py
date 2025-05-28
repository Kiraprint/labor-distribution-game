from sklearn.preprocessing import normalize
import numpy as np

def generate_embeddings(agent_data, embedding_dim=128):
    """
    Generate embeddings for agents based on their data.

    Parameters:
    - agent_data: A list of agent features (e.g., performance metrics, resource allocation).
    - embedding_dim: The dimension of the embedding space.

    Returns:
    - embeddings: A numpy array of shape (num_agents, embedding_dim) containing the embeddings.
    """
    num_agents = len(agent_data)
    embeddings = np.random.rand(num_agents, embedding_dim)
    embeddings = normalize(embeddings)  # Normalize the embeddings
    return embeddings

def update_embeddings(embeddings, updates):
    """
    Update the embeddings based on new information.

    Parameters:
    - embeddings: Current embeddings of the agents.
    - updates: A numpy array of shape (num_agents, embedding_dim) containing updates to the embeddings.

    Returns:
    - updated_embeddings: A numpy array of updated embeddings.
    """
    updated_embeddings = embeddings + updates
    updated_embeddings = normalize(updated_embeddings)  # Normalize after update
    return updated_embeddings

def compute_similarity(embedding_a, embedding_b):
    """
    Compute the cosine similarity between two embeddings.

    Parameters:
    - embedding_a: A numpy array representing the first embedding.
    - embedding_b: A numpy array representing the second embedding.

    Returns:
    - similarity: Cosine similarity between the two embeddings.
    """
    dot_product = np.dot(embedding_a, embedding_b)
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    similarity = dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0
    return similarity