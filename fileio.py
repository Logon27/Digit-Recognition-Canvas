import pickle
from network import Network

#Provides the ability to save and load neural network objects from files.

def saveNetwork(network: Network, filePath: str):
    with open(filePath, "wb") as file:
        print("Saving network to file {}...".format(filePath))
        pickle.dump(network, file, pickle.HIGHEST_PROTOCOL)
        print("Save complete.")

def loadNetwork(filePath: str) -> Network:
    with open(filePath, "rb") as file:
        print("Loading network from file {}...".format(filePath))
        network = pickle.load(file)
        print("Loading complete.")
        return network