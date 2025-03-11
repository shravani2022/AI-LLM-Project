import pygame
from pygame.math import Vector3
from OpenGL.GL import *
from OpenGL.GLU import *

class Object3D:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges

    def draw(self):
        # Draw vertices
        glPointSize(5)
        glBegin(GL_POINTS)
        for vertex in self.vertices:
            glVertex3fv(vertex)
        glEnd()

        # Draw edges
        glBegin(GL_LINES)
        for edge in self.edges:
            for vertex in edge:
                glVertex3fv(self.vertices[vertex])
        glEnd()

def main():
    # Initialize Pygame and OpenGL
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)

    # Cube vertices and edges
    vertices = [
        (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
        (1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1)
    ]
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # Bottom face
        (4,5), (5,6), (6,7), (7,4),  # Top face
        (0,4), (1,5), (2,6), (3,7)   # Connecting edges
    ]

    cube = Object3D(vertices, edges)

    # Set up perspective
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    # Rotation variables
    x_rotation = 0
    y_rotation = 0

    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen and reset the model view matrix
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5)

        # Rotate the cube
        glRotatef(x_rotation, 1, 0, 0)
        glRotatef(y_rotation, 0, 1, 0)

        # Set line and point colors
        glColor3f(0, 1, 1)  # Cyan color
        cube.draw()

        # Update rotation
        x_rotation += 1
        y_rotation += 1

        # Update display
        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()

if __name__ == "__main__":
    main()

# import numpy as np
# from sentence_transformers import SentenceTransformer
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

# class SentenceEmbeddingVisualizer:
#     def __init__(self, model_name='all-MiniLM-L6-v2'):
#         """
#         Initialize sentence embedding model
        
#         Parameters:
#         model_name (str): Pretrained sentence transformer model
#         """
#         self.model = SentenceTransformer(model_name)
    
#     def generate_embeddings(self, sentences):
#         """
#         Generate vector embeddings for sentences
        
#         Parameters:
#         sentences (list): List of sentences to embed
        
#         Returns:
#         numpy.ndarray: Vector embeddings
#         """
#         return self.model.encode(sentences)
    
#     def visualize_embeddings(self, sentences, method='tsne'):
#         """
#         Visualize sentence embeddings in 2D space
        
#         Parameters:
#         sentences (list): List of sentences to embed and visualize
#         method (str): Dimensionality reduction method ('tsne' or 'pca')
#         """
#         # Generate embeddings
#         embeddings = self.generate_embeddings(sentences)
        
#         # Dimensionality reduction
#         if method == 'tsne':
#             reducer = TSNE(n_components=2, random_state=42)
#         else:
#             reducer = PCA(n_components=2)
        
#         reduced_embeddings = reducer.fit_transform(embeddings)
        
#         # Plotting
#         plt.figure(figsize=(10, 8))
#         plt.scatter(
#             reduced_embeddings[:, 0], 
#             reduced_embeddings[:, 1], 
#             alpha=0.7
#         )
        
#         # Annotate points with sentences
#         for i, sentence in enumerate(sentences):
#             plt.annotate(
#                 sentence, 
#                 (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
#                 xytext=(5, 5),
#                 textcoords='offset points',
#                 fontsize=8,
#                 alpha=0.7
#             )
        
#         plt.title(f'Sentence Embeddings Visualization ({method.upper()})')
#         plt.xlabel('Dimension 1')
#         plt.ylabel('Dimension 2')
#         plt.tight_layout()
#         plt.show()
    
#     def calculate_similarity(self, sentences):
#         """
#         Calculate cosine similarity between sentence embeddings
        
#         Parameters:
#         sentences (list): List of sentences
        
#         Returns:
#         numpy.ndarray: Similarity matrix
#         """
#         embeddings = self.generate_embeddings(sentences)
#         similarity_matrix = np.dot(embeddings, embeddings.T)
#         return similarity_matrix

# # Example usage
# def main():
#     # Sample sentences
#     sentences = [
#         "I love machine learning",
#         "Deep learning is fascinating",
#         "Natural language processing is exciting",
#         "AI is transforming technology",
#         "Data science helps solve complex problems"
#     ]
    
#     # Create visualizer
#     visualizer = SentenceEmbeddingVisualizer()
    
#     # Visualize embeddings
#     visualizer.visualize_embeddings(sentences, method='tsne')
    
#     # Calculate and print similarity matrix
#     similarity = visualizer.calculate_similarity(sentences)
#     print("Similarity Matrix:")
#     print(similarity)

# if __name__ == "__main__":
#     main()