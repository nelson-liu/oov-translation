#!/usr/bin/perl -w

# Author: Ulf Hermjakob

$|=1;

use FindBin;
use Cwd "abs_path";
use File::Basename qw(dirname);
use File::Spec;

my $bin_dir = abs_path(dirname($0));
my $root_dir = File::Spec->catfile($bin_dir, File::Spec->updir());
my $data_dir = File::Spec->catfile($root_dir, "data");
my $lib_dir = File::Spec->catfile($root_dir, "lib");

use lib "$FindBin::Bin/../lib";
use List::Util qw(min max);
use NLP::utilities;
use NLP::stringDistance;
$util = NLP::utilities;
$sd = NLP::stringDistance;
$verbose = 0;
$print_stats_p = 0;

$cost_rule_filename = File::Spec->catfile($data_dir, "string-distance-cost-rules.txt");

$lang_code1 = "eng";
$lang_code2 = "eng";
%ht = ();
%sd_ht = ();

while (@ARGV) {
   $arg = shift @ARGV;
   if ($arg =~ /^-+lc1$/) {
      $lang_code_candidate = shift @ARGV;
      $lang_code1 = $lang_code_candidate if $lang_code_candidate =~ /^[a-z]{3,3}$/;
   } elsif ($arg =~ /^-+lc2$/) {
      $lang_code_candidate = shift @ARGV;
      $lang_code2 = $lang_code_candidate if $lang_code_candidate =~ /^[a-z]{3,3}$/;
   } elsif ($arg =~ /^-+(v|verbose)$/) {
      $verbose = 1;
   } else {
      print STDERR "Ignoring unrecognized arg $arg\n";
   }
}

print STDERR "lang_code1: $lang_code1 lang_code2: $lang_code2\n" if $verbose;

$sd->load_string_distance_data($cost_rule_filename, *ht, $verbose);
print STDERR "Loaded resources.\n" if $verbose;

my $chart_id = 0;
my $line_number = 0;
while (<>) {
   $line_number++;
   if ($verbose) {
      if ($line_number =~ /000$/) {
         if ($line_number =~ /0000$/) {
	    print STDERR $line_number;
         } else {
	    print STDERR ".";
         }
      }
   }
   my $line = $_;
   $line =~ s/^\xEF\xBB\xBF//;
   next if $line =~ /^\s*(\#.*)?$/;
   my $s1;
   my $s2;
   if (($s1, $s2) = ($line =~ /^(.*?)\t(.*?)\s*$/)) {
      $s1 = $util->dequote_string($s1);
      $s2 = $util->dequote_string($s2);
   } elsif ($line =~ /^\s*(#.*)$/) {
      print "\n";
   } else {
      print STDERR "Could not process line $line_number: $line" if $verbose;
      print "\n";
      next;
   }

   $cost = $sd->quick_romanized_string_distance_by_chart($s1, $s2, *ht, "", $lang_code1, $lang_code2);
   if ($verbose) {
      print STDERR "String distance(\"$s1\", \"$s1\", \"$s2\") = $cost l.$line_number\n";
   }
   print "$cost\n";
   $ht{N_STRING_PAIRS_AT_COST}->{$cost} = ($ht{N_STRING_PAIRS_AT_COST}->{$cost} || 0) + 1;
}
print STDERR "\n" if $verbose;

if ($print_stats_p) {
   my $total_cost = 0;
   my $acc_n = 0;
   my $n_pairs = 0;
   foreach $cost (sort { $a <=> $b } keys %{$ht{N_STRING_PAIRS_AT_COST}}) {
      $n = $ht{N_STRING_PAIRS_AT_COST}->{$cost};
      $acc_n += $n;
      $total_cost += $n * $cost;
      $n_pairs += $n;
      print "COST: $cost ($n/$acc_n)\n";
   }
   my $average_cost = ($n_pairs) ? ($total_cost / $n_pairs) : "n/a";
   print "TOTAL COST: $total_cost in $n_pairs string pairs (avg.: $average_cost)\n\n";

   foreach $chunk (sort { $sd_ht{N_COST_CHUNK}->{$b} <=> $sd_ht{N_COST_CHUNK}->{$a} }
		        keys %{$sd_ht{N_COST_CHUNK}}) {
      $n = $sd_ht{N_COST_CHUNK}->{$chunk};
      @line_numbers = sort { $a <=> $b } keys %{$sd_ht{EX_COST_CHUNK}->{$chunk}};
      @line_numbers = join(" ", @line_numbers[0..99], "...") if $#line_numbers >= 100;
      print "CHUNK: $chunk ($n) Ex.: @line_numbers\n";
   }
}

print STDERR "Lang-code-1: $lang_code1 Lang-code-2: $lang_code2\n" if $verbose;

exit 0;

