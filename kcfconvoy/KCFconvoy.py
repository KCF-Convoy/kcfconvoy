class KCFconvoy:
    def __init__(self):
        self.kcfvec_1 = kcf.KCFvec()

    def __call__(self, molecule, cpd_name=""):
        self.kcfvec_1.input_rdkmol(molecule, cpd_name=cpd_name)
        return {
            key: value["kegg_atom"]
            for key, value in self.kcfvec_1.kegg_atom_label.items()
        }
