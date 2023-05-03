from CGANTrainer import CGANTrainer

class SRCGAN(CGANTrainer):
    def __init__(self, args, network, dsc_network):
        super().__init__(args, network, dsc_network)
        self.patch_shape = (32, 32, 32)
        self.img_shape = (173, 207, 173)
        self.stride_shape = (16, 16, 16)