from sionna.phy import Object
import matplotlib.pyplot as plt
class DDGrid(Object):
    def __init__(self, delay_bins, doppler_bins, **kwargs):
        super().__init__(**kwargs)
        self.delay_bins = delay_bins   # M
        self.doppler_bins = doppler_bins  # N
        self.grid = None

    def collect(self, symbols):
        batch_dims = symbols.shape[:-1]
        self.grid = symbols.reshape(*batch_dims,
                                    self.doppler_bins,
                                    self.delay_bins).contiguous()
        return self.grid

    def show(self, batch_idx=0):
        data_to_plot = self.grid.abs() if self.grid.is_complex() else self.grid
        plot_view = data_to_plot[batch_idx].squeeze()

        plt.imshow(plot_view.cpu().numpy().T, aspect='auto', origin='lower')
        plt.colorbar(label='Magnitude')
        plt.xlabel('Doppler Bins')
        plt.ylabel('Delay Bins')
        plt.title('Delay-Doppler Grid')
        plt.show()