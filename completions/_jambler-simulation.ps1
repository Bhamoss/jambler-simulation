
using namespace System.Management.Automation
using namespace System.Management.Automation.Language

Register-ArgumentCompleter -Native -CommandName 'jambler-simulation' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)

    $commandElements = $commandAst.CommandElements
    $command = @(
        'jambler-simulation'
        for ($i = 1; $i -lt $commandElements.Count; $i++) {
            $element = $commandElements[$i]
            if ($element -isnot [StringConstantExpressionAst] -or
                $element.StringConstantType -ne [StringConstantType]::BareWord -or
                $element.Value.StartsWith('-')) {
                break
        }
        $element.Value
    }) -join ';'

    $completions = @(switch ($command) {
        'jambler-simulation' {
            [CompletionResult]::new('-t', 't', [CompletionResultType]::ParameterName, 'Which tasks to run')
            [CompletionResult]::new('--task', 'task', [CompletionResultType]::ParameterName, 'Which tasks to run')
            [CompletionResult]::new('-o', 'o', [CompletionResultType]::ParameterName, 'Output directory')
            [CompletionResult]::new('--output-dir', 'output-dir', [CompletionResultType]::ParameterName, 'Output directory')
            [CompletionResult]::new('-s', 's', [CompletionResultType]::ParameterName, 'The positive seed for the simulations. Random seed if not given')
            [CompletionResult]::new('--seed', 'seed', [CompletionResultType]::ParameterName, 'The positive seed for the simulations. Random seed if not given')
            [CompletionResult]::new('-c', 'c', [CompletionResultType]::ParameterName, 'The physical chance of capturing a packet when a sniffer is listening for it on a channel')
            [CompletionResult]::new('--capture-chance', 'capture-chance', [CompletionResultType]::ParameterName, 'The physical chance of capturing a packet when a sniffer is listening for it on a channel')
            [CompletionResult]::new('-n', 'n', [CompletionResultType]::ParameterName, 'The number of sniffers')
            [CompletionResult]::new('--nb-sniffers', 'nb-sniffers', [CompletionResultType]::ParameterName, 'The number of sniffers')
            [CompletionResult]::new('-h', 'h', [CompletionResultType]::ParameterName, 'Prints help information')
            [CompletionResult]::new('--help', 'help', [CompletionResultType]::ParameterName, 'Prints help information')
            [CompletionResult]::new('-V', 'V', [CompletionResultType]::ParameterName, 'Prints version information')
            [CompletionResult]::new('--version', 'version', [CompletionResultType]::ParameterName, 'Prints version information')
            break
        }
    })

    $completions.Where{ $_.CompletionText -like "$wordToComplete*" } |
        Sort-Object -Property ListItemText
}
