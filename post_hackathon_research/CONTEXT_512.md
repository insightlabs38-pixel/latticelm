# Context-length gate

The full eight-task 4K prompt audit found a maximum meaningful truncation fraction of 23.90% at context 256.
Decision: run 512-token context — audit and time gate passed.
The 512 configuration uses batch 16 × context 512, preserving the 8192-token effective optimization batch of the 256 control (batch 32 × context 256). Tokenized document chunks differ by context length, as intended.

WikiText BPB changed from 1.62941 to 1.63626; source-wise held-out losses and eight-task capability samples are in the run archive.
