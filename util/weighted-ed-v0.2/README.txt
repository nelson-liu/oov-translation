Script: weighted-ed.pl (weighted string edit distance)
Version: 0.2 (August 16, 2017)
Written by: Ulf Hermjakob, USC/ISI

Usage: bin/weighted-ed.pl [-lc1 <iso-lang-code1>] [-lc2 <iso-lang-code2>] [-v|-verbose] < STDIN > STDOUT
Example: bin/weighted-ed.pl < text/sample-pairs-01.txt
Example: bin/weighted-ed.pl -lc1 orm -lc2 orm < text/oromo-01.txt

Input is a file of tab-separated pairs of strings to be compared.
Output is edit distance between the strings (a number).

Change log
v0.2 Added 33 rules for Oromo, 9 rules for Tigrinya; mostly about morphological variation such as plural forms.

Notes
(1) Comparison is case insensitive.
(2) For deficient input lines (e.g. no tab on line), the script returns an empty line.
(3) Default score for any addition/deletion is 1.
(4) Currently, a character replacement has a default score of 2 (1 addition + 1 deletion).
(5) Default language is English (lang-code "eng").
(6) Currently no Oromo or Tigrinya-specific rules. But universal rules apply.
