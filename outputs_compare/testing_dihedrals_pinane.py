import collections

series1 = [
    'C3-C1-C2-C4', 'C3-C1-C2-C5', 'C3-C1-C2-H10', 'C8-C1-C2-C4', 'C8-C1-C2-C5', 'C8-C1-C2-H10',
    'C9-C1-C2-C4', 'C9-C1-C2-C5', 'C9-C1-C2-H10', 'C2-C1-C3-C4', 'C2-C1-C3-C6', 'C2-C1-C3-H11',
    'C8-C1-C3-C4', 'C8-C1-C3-C6', 'C8-C1-C3-H11', 'C9-C1-C3-C4', 'C9-C1-C3-C6', 'C9-C1-C3-H11',
    'C2-C1-C8-H20', 'C2-C1-C8-H21', 'C2-C1-C8-H22', 'C3-C1-C8-H20', 'C3-C1-C8-H21', 'C3-C1-C8-H22',
    'C9-C1-C8-H20', 'C9-C1-C8-H21', 'C9-C1-C8-H22', 'C2-C1-C9-H23', 'C2-C1-C9-H24', 'C2-C1-C9-H25',
    'C3-C1-C9-H23', 'C3-C1-C9-H24', 'C3-C1-C9-H25', 'C8-C1-C9-H23', 'C8-C1-C9-H24', 'C8-C1-C9-H25',
    'C1-C2-C4-C3', 'C1-C2-C4-H12', 'C1-C2-C4-H13', 'C5-C2-C4-C3', 'C5-C2-C4-H12', 'C5-C2-C4-H13',
    'H10-C2-C4-C3', 'H10-C2-C4-H12', 'H10-C2-C4-H13', 'C1-C2-C5-C7', 'C1-C2-C5-H14', 'C1-C2-C5-H15',
    'C4-C2-C5-C7', 'C4-C2-C5-H14', 'C4-C2-C5-H15', 'H10-C2-C5-C7', 'H10-C2-C5-H14', 'H10-C2-C5-H15',
    'C1-C3-C4-C2', 'C1-C3-C4-H12', 'C1-C3-C4-H13', 'C6-C3-C4-C2', 'C6-C3-C4-H12', 'C6-C3-C4-H13',
    'H11-C3-C4-C2', 'H11-C3-C4-H12', 'H11-C3-C4-H13', 'C1-C3-C6-C7', 'C1-C3-C6-H16', 'C1-C3-C6-H17',
    'C4-C3-C6-C7', 'C4-C3-C6-H16', 'C4-C3-C6-H17', 'H11-C3-C6-C7', 'H11-C3-C6-H16', 'H11-C3-C6-H17',
    'C2-C5-C7-C6', 'C2-C5-C7-H18', 'C2-C5-C7-H19', 'H14-C5-C7-C6', 'H14-C5-C7-H18', 'H14-C5-C7-H19',
    'H15-C5-C7-C6', 'H15-C5-C7-H18', 'H15-C5-C7-H19', 'C3-C6-C7-C5', 'C3-C6-C7-H18', 'C3-C6-C7-H19',
    'H16-C6-C7-C5', 'H16-C6-C7-H18', 'H16-C6-C7-H19', 'H17-C6-C7-C5', 'H17-C6-C7-H18', 'H17-C6-C7-H19'
]

series2 = [
    'C3-C5-C6-C7', 'C5-C7-H14-H19', 'C1-C3-C8-H21', 'C1-C3-C9-H23', 'C1-C2-C9-H23', 'C1-C2-C5-C8',
    'C5-C7-H15-H18', 'C1-C3-C8-H11', 'C1-C2-C4-H13', 'C2-C4-C5-H12', 'C1-C3-C6-H16', 'C2-C5-C7-H18',
    'C2-C4-C5-C7', 'C1-C2-C5-C7', 'C2-C5-C7-H10', 'C1-C2-C8-H22', 'C5-C7-H14-H18', 'C1-C2-C4-C8',
    'C1-C3-C8-H22', 'C2-C4-C5-H15', 'C1-C8-C9-H23', 'C1-C2-C5-H15', 'C1-C2-C4-H12', 'C1-C3-C4-H12',
    'C1-C3-C4-C9', 'C1-C8-C9-H20', 'C2-C4-C5-H13', 'C5-C6-C7-H17', 'C3-C6-H11-H16', 'C3-C4-H11-H12',
    'C2-C5-C6-C7', 'C1-C3-C9-H11', 'C1-C3-C4-H13', 'C2-C5-H10-H15', 'C1-C8-C9-H24', 'C3-C4-C6-H12',
    'C1-C2-C3-C5', 'C6-C7-H16-H19', 'C1-C3-C4-C8', 'C3-C4-C6-H17', 'C1-C3-C6-C9', 'C1-C8-C9-H21',
    'C3-C4-H11-H13', 'C3-C4-C6-C7', 'C6-C7-H17-H19', 'C1-C2-C3-C6', 'C1-C2-C8-H20', 'C1-C3-C9-H24',
    'C1-C2-C9-H24', 'C1-C2-C9-H10', 'C1-C2-C3-H11', 'C3-C4-C6-H13', 'C1-C8-C9-H22', 'C6-C7-H16-H18',
    'C1-C3-C9-H25', 'C5-C6-C7-H15', 'C1-C2-C9-H25', 'C2-C4-H10-H12', 'C2-C3-C4-C6', 'C3-C6-C7-H19',
    'C2-C5-H10-H14', 'C3-C6-C7-H11', 'C1-C3-C6-C8', 'C5-C6-C7-H16', 'C6-C7-H17-H18', 'C2-C4-C5-H14',
    'C1-C3-C6-H17', 'C1-C2-C5-H14', 'C1-C2-C3-C4', 'C1-C3-C6-C7', 'C1-C8-C9-H25', 'C1-C2-C8-H21',
    'C1-C2-C3-H10', 'C2-C4-H10-H13', 'C1-C3-C8-H20', 'C2-C3-C4-C5', 'C3-C6-C7-H18', 'C3-C4-C6-H16',
    'C1-C2-C5-C9', 'C5-C6-C7-H14', 'C2-C3-C4-H10', 'C3-C6-H11-H17', 'C5-C7-H15-H19', 'C1-C2-C4-C9',
    'C2-C3-C4-H11', 'C2-C5-C7-H19', 'C1-C2-C8-H10'
]

#now, let's find the intersection of the two series
#but we need to take into account that the order of the atoms in the dihedral angle is not important
#get length of series1 and series2
print("length of series1",len(series1))
print("length of series2",len(series2)
)
#first, let's create a function that will sort the atoms in the dihedral angle
def sort_dihedral(dihedral):
    atoms = dihedral.split('-')
    atoms.sort()
    return '-'.join(atoms)

#now, let's sort the dihedrals in the two series
sorted_series1 = [sort_dihedral(dihedral) for dihedral in series1]
sorted_series2 = [sort_dihedral(dihedral) for dihedral in series2]
print("sorted_series1",set(sorted_series1))
print("sorted_series2",set(sorted_series2))

#length of sorted

print("length of sorted_series1",len(set(sorted_series1)))
print("length of sorted_series2",len(set(sorted_series2)))
#length
# print("length of sorted_series1",len(sorted_series1))
# print("length of sorted_series2",len(sorted_series2))
#now, let's find elements that are in series 1 but not in series 2

# only_in_series1 = sorted_series1 - sorted_series2
only_in_series2 = set(sorted_series2) - set(sorted_series1)
# print("only_in_series1",only_in_series1)
print("only_in_series2",only_in_series2)

#let's get to know the 3 elements in only_in_series1
#only_in_series1 = {'C1-C2-C3-C4', 'C1-C2-C3-C5', 'C1-C2-C3-H11'}

#get repeated elements in series1
repeated_elements = [item for item, count in collections.Counter(sorted_series1).items() if count > 1]
print("repeated_elements",repeated_elements)

