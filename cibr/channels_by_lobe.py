import sys

import mne

print "Included selections are: "
print ' Vertex'
print ' Left-temporal'
print ' Right-temporal'
print ' Left-parietal'
print ' Right-parietal'
print ' Left-occipital'
print ' Right-occipital'
print ' Left-frontal'
print ' Right-frontal'

input_ = raw_input("Write here a combination of them (e.g `Right-temporal` or `Left-occipital Right-occipital` without the quotes): ")
selections =  input_.split(' ')

channels = mne.selection.read_selection(selections)

channel_str = ' '.join([ch[-4:] for ch in channels])
print channel_str

if len(sys.argv) > 1:
    with open(sys.argv[-1], 'wb') as f:
        f.write(channel_str)
