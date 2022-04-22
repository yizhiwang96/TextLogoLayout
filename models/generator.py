from models.img_encoder import ImageEncoder

class Generator(nn.Module):

    def __init__(self, input_size, output_size, max_seqlen):
        """
        Parameters:
        """
        self.img_encoder =
        self.text_encoder =
        self.

    def forward(self, imgs, text_embeds, seq_len):
        """
        Parameters:
        imgs: [self.batch_size, 3, self.input_size, self.max_seqlen * self.input_size]
        coords: [self.batch_size, self.max_seqlen, 4]
        seq_len: [self.batch_size, 1]
        """

        # prepare the coordinates (0, 1) -> (0, output_size)
        coords = coords * self.output_size
        coords = coords.reshape(self.batch_size * self.max_seqlen, 4)

        # the predicted geometric paramters: centre points, width, height
        xc = coords[:, 0:1]
        yc = coords[:, 1:2]
        w = coords[:, 2:3]
        h = coords[:, 3:4]
        seq_mask = sequence_mask(seq_len, maxlen=self.max_seqlen)