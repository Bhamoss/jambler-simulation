complete -c jambler-simulation -s t -l task -d 'Which tasks to run' -r -f -a "all channel-occurrences channel-recovery connection-interval full"
complete -c jambler-simulation -s o -l output-dir -d 'Output directory' -r
complete -c jambler-simulation -s s -l seed -d 'The positive seed for the simulations. Random seed if not given' -r
complete -c jambler-simulation -s c -l capture-chance -d 'The physical chance of capturing a packet when a sniffer is listening for it on a channel' -r
complete -c jambler-simulation -s n -l nb-sniffers -d 'The number of sniffers' -r
complete -c jambler-simulation -s h -l help -d 'Prints help information'
complete -c jambler-simulation -s V -l version -d 'Prints version information'
