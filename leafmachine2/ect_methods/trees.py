from Bio import Phylo
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

class PhylogeneticTreeViewer:
    def __init__(self, tree_file):
        self.tree_file = tree_file
        self.tree_biopython = None

    def load_tree_biopython(self):
        print("Loading tree with Biopython...")
        with open(self.tree_file, 'r') as f:
            self.tree_biopython = Phylo.read(f, 'newick')
        print("Tree loaded successfully with Biopython.")

    def plot_circular_phylogeny_biopython(self):
        if self.tree_biopython:
            print("Displaying circular phylogeny (Biopython)...")
            fig = plt.figure(figsize=(12, 12))  # Increase the figure size to fit a larger tree
            ax = fig.add_subplot(1, 1, 1)
            
            # Create FontProperties for smaller labels
            font_properties = FontProperties(size=1)  # Adjust size as needed

            # Biopython's draw method with circular orientation
            Phylo.draw(self.tree_biopython, axes=ax, do_show=False)
            
            # Loop through labels and apply font properties
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(font_properties)

            ax.set_aspect('equal', 'box')
            plt.show(block=True) 
        else:
            print("Tree not loaded with Biopython. Please load the tree first.")

    def plot_tree_biopython(self):
        if self.tree_biopython:
            print("Displaying tree using matplotlib (Biopython)...")
            fig = plt.figure(figsize=(30, 30))  # Increase figure size to better fit a large tree
            ax = fig.add_subplot(1, 1, 1)
            
            # Create FontProperties for smaller labels
            font_properties = FontProperties(size=1)  # Adjust size as needed

            # Plot the tree with Biopython's draw method
            Phylo.draw(self.tree_biopython, axes=ax, do_show=False)

            # Apply font properties to the labels
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(font_properties)

            ax.set_aspect('auto')
            plt.show(block=True)
        else:
            print("Tree not loaded with Biopython. Please load the tree first.")


if __name__ == "__main__":
    # tree_file = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/Cornales_41934.tre.gz"
    tree_file = "D:/T_Downloads/URT_dated"
    viewer = PhylogeneticTreeViewer(tree_file)

    viewer.load_tree_biopython()
    viewer.plot_circular_phylogeny_biopython()
    viewer.plot_tree_biopython()
