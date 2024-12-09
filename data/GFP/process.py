import copy

protein_name = 'avGFP'

with open(f'{protein_name}.txt', 'r') as f_in:
    sequence = f_in.readline().strip()

scores = []
with open(f'{protein_name}_score.txt', 'r') as f_in:
    for line in f_in:
        scores.append(float(line.strip()))

mutants = []
with open(f'{protein_name}_mutant.txt', 'r') as f_in:
    for line in f_in:
        mutants.append(line.strip())

with open(f'{protein_name}.csv', 'w') as f_out:
    f_out.write('mutant,mutated_sequence,score,split\n')
    for idx, (score, mutant) in enumerate(zip(scores, mutants)):
        mutant_sequence = copy.deepcopy(sequence)
        output_mutant = []
        if '*' in mutant or '.' in mutant or 'WT' in mutant:
            continue
        for m in mutant.split(':'):
            wt_aa, mt_aa, pos = m[0], m[-1], int(m[1:-1])
            if mutant_sequence[pos] == wt_aa:
                mutant_sequence = mutant_sequence[:pos] + mt_aa + mutant_sequence[pos+1:]
            else:
                import ipdb; ipdb.set_trace()
            output_mutant.append(wt_aa + str(pos+1) + mt_aa)
        if idx % 10 == 0:
            f_out.write(f'{":".join(output_mutant)},{mutant_sequence},{score},{2}\n')
        else:
            f_out.write(f'{":".join(output_mutant)},{mutant_sequence},{score},{0}\n')

