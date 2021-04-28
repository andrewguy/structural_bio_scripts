'''A collection of python functions for dealing with MD data in MDTraj.

Includes functions for identification of dihedral angles in carbohydrates.

The function `transfer_atomic_coordinates` is useful when setting up
an MD simulation and transferring coordinates from a crystal structure to
a glycan structure from Glycam-Web (which contains the correct labelling for 
the Glycam forcefield)
'''

import mdtraj as md
import numpy as np
import networkx as nx


dihedrals = {'phi': ('O5', 'C1', 'O6', 'C6'),
             'psi': ('C1', 'O6', 'C6', 'C5'),
             'omega': ('O6', 'C6', 'C5', 'C4')
            }

def get_atom_type(atom_name):
    '''Return the atom type/label from the full atom name.

    For example, with 'ROH1-O1', returns 'O1'.
    '''
    return str(atom_name).split('-')[-1]


def _get_specific_neighbour(g, node, atom_type):
    potential_neighbours = g.neighbors(node)
    for neighbour in potential_neighbours:
        if get_atom_type(neighbour) == atom_type:
            return neighbour
    return None


def retrieve_dihedrals(glycan_nodes, bondgraph, dihedral_type='phi', dihedral_defs=dihedrals):
    '''Retrieve all possible dihedral bonds based on given dihedral atom types.

    Args:
        glycan_nodes (list): A list of all glycan nodes in the bondgraph.
        bondgraph (Graph): A networkx bondgraph.
        dihedral_type (str): A key for the `dihedral_defs dictionary.
        dihedral_defs (dict): A dictionary of dihedral types. Each entry is a
            4-tuple, and defines the atom types for the given dihedral, or each entry is
            a list of possible 4-tuples that define a possible dihedral.
    Returns:
        A list of lists, with each list containing the four atom objects for each dihedral.
    '''
    dihedrals = []
    dihedral_def_list = dihedral_defs[dihedral_type]
    # Maintain compatibility with previous version of function
    if isinstance(dihedral_def_list[0], str):
        dihedral_def_list = [dihedral_def_list,]
    for dihedral_def in dihedral_def_list:
        for node in glycan_nodes:
            next_node = None
            possible_dihedral = []
            for diname in dihedral_def:
                if next_node is not None:
                    next_node = _get_specific_neighbour(bondgraph, next_node, diname)
                else:
                    next_node = node
                if get_atom_type(next_node) == diname:
                    possible_dihedral.append(next_node)
                else:
                    break
            if len(possible_dihedral) == len(dihedral_def):
                dihedrals.append(possible_dihedral)
    return dihedrals

class GraphMatcherForPDBStructures(nx.isomorphism.GraphMatcher):
    '''A subclass of nx.isomorphism.GraphMatcher.

    Allows matching of pdb structures as represented by NetworkX graphs.

    Useful when you want to transfer coordinates from a reference pdb file to a
    destination pdb file, for example when building a carbohydrate with GLYCAM-WEB
    and transferring the carbohydrate coordinates from a crystal structure.
    '''
    def __init__(self, G1, G2):
        """
        Convert node labels to integers. Store old labels in 'label' attribute.
        """
        g1 = nx.convert_node_labels_to_integers(G1, label_attribute='label')
        g2 = nx.convert_node_labels_to_integers(G2, label_attribute='label')
        super(GraphMatcherForPDBStructures, self).__init__(g1, g2)


    def atom_mapping(self):
        '''Return a mapping of atom objects between two graphs.

        Explicitly checks that element types are the same between
        each matching node.
        '''
        # Note: Need to call self.is_isomorphic before mapping is defined.
        # Good time to check that a suitable mapping has been found.
        if not self.is_isomorphic():
            raise Warning("No mapping between structure found!")
        g1_node_labels = nx.get_node_attributes(self.G1, 'label')
        g2_node_labels = nx.get_node_attributes(self.G2, 'label')
        mapping = self.mapping
        return {g1_node_labels[k]: g2_node_labels[v] for k, v in mapping.items()}


    def semantic_feasibility(self, G1_node, G2_node):
        '''Match element types between nodes when checking for isomorphism.'''
        # Note: Could check for other attributes here if needed, or even check
        # correspondence of a few key atom labels. Might be needed for structures with
        # a level of symmetry.
        g1_node_ele = nx.get_node_attributes(self.G1, 'label')[G1_node].element.atomic_number
        g2_node_ele = nx.get_node_attributes(self.G2, 'label')[G2_node].element.atomic_number
        if g1_node_ele == g2_node_ele:
            return True
        else:
            return False


def transfer_atomic_coordinates(reference_pdb, destination_pdb, output_filename='new_coordinates.pdb',
                                output_format='pdb'):
    '''Transfer atomic coordinates from a reference pdb file to destination pdb file.

    PDB files must have all bonds defined in CONECT records.
    If these are not defined, can use a tool such as `mol_connect` in the `Silico` Perl molecular
    modelling toolkit to write likely connections.

    Saves a new PDB file with the same atom labels as the destination pdb file, but with coordinates
    from the reference pdb file.
    '''
    # Read in files and get bond graphs
    dest_structure = md.load_pdb(destination_pdb)
    ref_structure = md.load_pdb(reference_pdb)
    dest_bond_graph = dest_structure.topology.to_bondgraph()
    ref_bond_graph = ref_structure.topology.to_bondgraph()

    # Generate mapping between atoms for one possible isomorphism.
    GM = GraphMatcherForPDBStructures(dest_bond_graph, ref_bond_graph)
    GM.atom_mapping()

    # Transfer xyz coordinates from reference structure to destination structure.
    dest_atom_list = list(dest_structure.topology.atoms)
    ref_atom_list = list(ref_structure.topology.atoms)
    ref_xyz = ref_structure.xyz
    new_xyz = [ref_xyz[0][ref_atom_list.index(GM.atom_mapping()[dest_atom])] for dest_atom in dest_atom_list]
    dest_structure.xyz = np.array(new_xyz)
    if output_format == 'pdb':
        dest_structure.save_pdb(output_filename)
    elif output_format == 'gro':
        dest_structure.save_gro(output_filename)
    else:
        raise ValueError("Not a valid output format.")
    return
