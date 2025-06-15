from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='load model checkpoint')
        parser.add_argument('--train_opt_path', type=str, default='./train_opt.txt', help='Path to the training options text file (used to load model architecture).')
        parser.add_argument('--input_nii_filename', type=str, default='r_FBB_coreg.nii', help='Filename of the input NII image within each patient directory.')
        parser.add_argument('--output_filename_suffix', type=str, default='_gen.nii', help='Suffix for the generated NII file.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        # Set the default = 5000 to test the whole test set.
        parser.add_argument('--num_test', type=int, default=5000, help='how many test images to run')

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
