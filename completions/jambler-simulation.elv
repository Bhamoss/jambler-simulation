
use builtin;
use str;

edit:completion:arg-completer[jambler-simulation] = [@words]{
    fn spaces [n]{
        builtin:repeat $n ' ' | str:join ''
    }
    fn cand [text desc]{
        edit:complex-candidate $text &display-suffix=' '(spaces (- 14 (wcswidth $text)))$desc
    }
    command = 'jambler-simulation'
    for word $words[1..-1] {
        if (str:has-prefix $word '-') {
            break
        }
        command = $command';'$word
    }
    completions = [
        &'jambler-simulation'= {
            cand -t 'Which tasks to run'
            cand --task 'Which tasks to run'
            cand -o 'Output directory'
            cand --output-dir 'Output directory'
            cand -s 'The positive seed for the simulations. Random seed if not given'
            cand --seed 'The positive seed for the simulations. Random seed if not given'
            cand -c 'The physical chance of capturing a packet when a sniffer is listening for it on a channel'
            cand --capture-chance 'The physical chance of capturing a packet when a sniffer is listening for it on a channel'
            cand -n 'The number of sniffers'
            cand --nb-sniffers 'The number of sniffers'
            cand -h 'Prints help information'
            cand --help 'Prints help information'
            cand -V 'Prints version information'
            cand --version 'Prints version information'
        }
    ]
    $completions[$command]
}
