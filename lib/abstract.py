class ComponentData(object):
    """
    """
    def __init__(self, source_stft, sensor_stft, sensor_topo, source_psd, freqs, length, info):
        self.source_stft = source_stft
        self.sensor_stft = sensor_stft
        self.source_psd = source_psd
        self.sensor_topo = sensor_topo
        self.freqs = freqs
        self.info = info
        self.length = length

