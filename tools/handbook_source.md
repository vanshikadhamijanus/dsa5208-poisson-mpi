---
title: "Filling cloud gaps in satellite sea temperature with a parallel Poisson solver: a beginner's handbook"
subtitle: "Every method, file, number and figure in the DSA5208 Project 3 solver, explained from zero"
author: "Aditya Vijindra Pandhari, Apurwa Kumari, Vanshika Dhamija"
date: "November 2026"
---

# Preface

This handbook accompanies the project `poisson-mpi/`, written for DSA5208 Project 3 at the National University of Singapore: a parallel solver for the Poisson equation, validated on a textbook problem and then used to fill the cloud holes in a year of daily satellite sea-surface temperature maps of the seas around India. It is written for a smart reader with no background in numerical methods, parallel computing, satellite data or C++. Every technical term is explained the first time it appears in a Part, with an everyday analogy, and every number is quoted from a file in the project folder.

Each section after Part A answers the same five questions in the same order, marked in bold: What is it? Why did we do it? How does it work? What came out of it? How would I explain it to a classmate in two sentences? If you only have a minute for a section, read the last of the five.

Two honesty notes apply throughout. First, all timing numbers come from one laptop (an Apple M4 Max with 12 performance and 4 efficiency cores). The 256-core and GPU measurements that the assignment asks for have not been run yet. The job scripts for the NUS Atlas and Vanda clusters exist, and the analysis regenerates every figure from their output, but until then the figures are laptop figures and are labelled as such. Second, the CUDA (GPU) version of the code has been compiled and run only through a CPU emulation layer that checks its logic. It has never run on a GPU.

The handbook has six Parts. Part A is the big picture. Part B teaches the methods from scratch. Part C is about the data: the satellite dataset and where it comes from, what goes into a run, what comes out, and every assumption. Part D walks through the folder and every file of code. Part E explains every figure and table. Part F is for presenting and defending the work: a glossary, thirty-six likely questions with answers, a presentation script, and a procedure for tracing any number back to its source.

# Part A. The big picture

This Part is the view from the hill before the walk through the valley. It says what the project is about, why anyone would choose this topic, who cares about the answer, and how the written report is laid out. Every later Part assumes you have read this one. Technical words are explained as they arrive, each with a one-line definition and an everyday comparison, and the same words are collected in the glossary in Part F.

## A1. The one-page summary

**What is it?** Take a thin square metal plate. You hold its four edges at a fixed cold temperature and you warm the middle with a heater whose strength varies from place to place. After a while the plate settles: every point ends up at a steady temperature that depends on the heater and on the cold edges. The question "what is the temperature at every point once the plate has settled?" is the Poisson problem. It is a sentence in mathematics, not in physics alone: the same question describes a soap film stretched over a wire, the electric field around charges, and the pressure inside flowing water. We solved it on a square (the plate) and on a cube (a solid block), and we solved it on one computer core, then on many cores working together.

Then we pointed the same solver at a real problem. Satellites photograph the temperature of the sea surface every day, but their infrared cameras cannot see through clouds, so every daily map has holes, and over the seas around India during the monsoon the holes can cover most of the map. A hole can be filled by asking for the smoothest temperature field that agrees with the clear pixels around it. That request is the Poisson problem again, with the heater switched off: each cloudy pixel should be the average of its neighbours, and the clear pixels play the part of the fixed plate edges. We did this for one day at a time (2D: a hole is filled from the clear sea around it) and for a whole year stacked into a block with time as the third direction (3D: a hole is also filled from the same place yesterday and tomorrow). The data are 365 daily maps of 801 by 534 pixels (about 4 km each) covering the Arabian Sea and the Bay of Bengal in 2025, downloaded from NOAA.

**Why did we do it?** The project for this course asked for an algorithm that is hard to share among many computers, implemented once the slow way and then several fast ways, with measurements of how the speed changes as more computers join in. The settled-plate problem fits that brief exactly. The plate is cut into a grid of points, like the squares on graph paper. The temperature at any one point is simply the average of its four neighbours plus a nudge from the heater. That sounds easy, and it is, but every point needs its neighbours, so when the grid is cut into pieces and handed to different computers, each computer keeps needing values that belong to its neighbour. The computers must keep talking. That talking, and how to make it cheap, is the whole study. The satellite maps add what a textbook problem cannot: a real dataset with coasts and with holes of every shape, a question people actually ask (what was the sea temperature under the clouds?), and a way to check the answer by hiding pixels we know and seeing how well we get them back.

**How does it work?** The program starts with a guess (every point at zero) and repeats a simple update, point by point, until the numbers stop changing. One update of the whole grid is called an iteration, like one stir of a pot. Two recipes for the update were written. The first, named after the mathematician Jacobi, is the plain "replace me with the average of my neighbours" rule. It is simple and needs many thousands of stirs. The second, conjugate gradient, is cleverer about the direction of each change and needs only a few hundred. To run on many cores the grid is cut into rectangular blocks, one per worker, and after each stir every worker posts its edge rows to its neighbours, the way people in adjacent allotments pass tools over the fence. Each worker is a separate program called a rank, the workers talk through a standard called MPI (Message Passing Interface), and inside each worker the loops can be shared among threads (several hands in one kitchen) or sent to a graphics card.

**What came out of it?** Six things, all checked by running the code rather than argued on paper. First, the answer is right: when the grid is made twice as fine, the gap between the computed temperature and the exact one shrinks by a factor of four, measured as 3.88, 3.94, 3.97, 3.98 and 3.99 across five doublings. A factor of four is what the theory promises. Second, every version of the program agrees: serial, threaded, MPI in four variants, hybrid and the graphics-card version run through an emulator all produce the same errors to about fifteen significant digits (the largest difference was 3.2 in the fifteenth decimal place). Third, sharing the work pays: on a 1024 by 1024 plate the conjugate gradient solver took 3.73 seconds on one core and 0.46 seconds on twelve, which is 8.3 times faster, a 70 percent return on the extra cores. Fourth, more is not always better: using all sixteen cores of the laptop was slower than twelve (1.04 seconds against 0.46), because four of the sixteen are low-power cores that finish late and the other twelve stand waiting at the fence. Fifth, the recipe details matter: writing straight into a neighbour's memory (a method called one-sided communication) was between two and eighty times slower than ordinary messages on this laptop, checking convergence after every single stir cost 2.6 times the total run, and eight workers with two threads each beat sixteen workers with one thread in every test. Sixth, the satellite holes can be filled, and the year helps only for the big ones: hiding 64 km squares of clear pixels on a cloudy April day and filling them from the rest of the day gives an error of 0.15 degrees, against 0.92 degrees for guessing the mean temperature, and the maps are complete. Stacking the whole year made small holes slightly worse (consecutive days differ more than neighbouring pixels do) but 250 km holes better (0.29 against 0.38 degrees), and it brought back the eddies and fronts under the clouds that the one-day fill smooths away.

**How would I explain it to a classmate in two sentences?** We computed the steady temperature of a plate and a block by repeatedly averaging each point with its neighbours, taught the program to run across many processors that trade edge rows after every round, and then used it to fill the cloud holes in a year of satellite sea-temperature maps around India, checking the fill by hiding pixels we knew. The answer is correct to the precision the maths allows, the program runs about eight times faster on twelve cores, the filled maps beat the trivial guess by a wide margin, and the runs on 256 cluster cores and on a real graphics card are scripted but still waiting to be executed on the university machines.

## A2. Why this topic

**What is it?** The Poisson equation is the simplest partial differential equation that still matters. A differential equation is a rule about how a quantity changes from place to place rather than a formula for the quantity itself. A recipe that says "each step of the staircase is 18 centimetres higher than the last" is a differential equation for the height of the staircase. Poisson's version says how the curvature of a field (how much it bends) is set by a source. Its grid form is a stencil: a fixed pattern of neighbours, like a cookie cutter, applied at every point. The two solvers (Jacobi and conjugate gradient) are the first two methods any numerical methods course teaches for such systems.

**Why did we do it?** The lecturer set a clear condition in lecture 7: the simpler the algorithm, the more sophisticated its parallelisation must be. A stencil update is about ten lines of code, so the sophistication had to come from everywhere else. This project answers the condition in six ways. It implements four different strategies for passing edge rows between workers (blocking messages, non-blocking messages, non-blocking messages with computation hidden underneath, and one-sided writes into a neighbour's memory). It tests different shapes of the cut (long strips versus square tiles versus cubes). It measures how much a global convergence check costs and how often one can afford it. It tries every split of sixteen cores between MPI workers and threads. It ports the kernels to a graphics card. And it does all of this in both two and three dimensions, as the lecture 12 feedback asked, from one shared source so that the comparisons are fair.

**How does it work?** The reason the problem is a good test is that it sits between two extremes. An embarrassingly parallel task is one where the workers never need to speak, like a hundred people each painting their own fence. Splitting it gives a perfect speedup with no skill. A problem where every point depends on every other point cannot be split at all. The stencil is in between: each point depends only on its immediate neighbours, so a worker owning a block needs only one layer of foreign values around its edge. That layer is the halo, a ring of borrowed cells, like the margin of a map page that repeats a strip of the next page so you can see where the road goes. The size of the halo relative to the block decides how much talking there is. A 512 by 512 block has 262144 cells but a halo of only 2048, under one percent. That ratio is why the method can scale, and the wait for late neighbours is why it eventually stops scaling.

**What came out of it?** The topic delivered on every rubric item. The serial study shows the time per iteration growing linearly with the number of grid points (slopes of 1.05, 1.01, 1.01 and 1.00 on log-log axes for the four solver and dimension combinations). The MPI study shows good efficiency up to 12 cores for conjugate gradient (70 percent in 2D, 55 percent in 3D) and lower for Jacobi (40 and 28 percent), with the reason visible in the timers: Jacobi is limited by memory speed rather than arithmetic, so extra cores sharing one memory system gain less. The design experiments gave clear winners and one clear loser. The hybrid study gave a practical rule for this machine: two threads per rank. The graphics-card port was validated for correctness but not for speed, since no GPU was available.

A word on why the application is satellite sea temperature and not something else. The course is run by the Mathematics department and two of us study data science for sustainability, so we wanted a dataset where the parallel solver answers a real environmental question rather than decorating a benchmark. Filling cloud gaps in sea-surface temperature is a textbook use of the Laplace equation (image-processing people call it harmonic in-painting), it is a baseline against which the operational methods are judged, the data are free and daily, and the seas around India are among the cloudiest on Earth in the monsoon months, so the holes are real and large. It also gives the 3D solver a meaning: the third axis is time, and the question "does knowing yesterday and tomorrow help fill today?" has a measurable answer.

**How would I explain it to a classmate in two sentences?** The Poisson stencil is simple enough to write in an afternoon but needs neighbours at every step, so it forces you to deal with communication, which is the real subject of parallel computing. We used that simplicity to spend the project on the hard part, how to cut, how to talk, how often to synchronise, and how to split a node between processes and threads, and then on a real question, what the sea temperature was under the clouds.

## A3. Who the decision-makers and stakeholders are and what decision we are helping them make

**What is it?** A stakeholder is anyone whose outcome depends on this work, and a decision-maker is a stakeholder who must choose something because of it. For a student project the list is short and honest: the students, the grader, an engineer who might reuse the findings, a scientist who needs complete daily sea-temperature maps, and a reader who wants to learn. Each has a different decision in front of them and needs a different slice of the evidence. The scientist's decision is whether a plain Laplace fill is good enough for their purpose (a heat-wave count, a coral-bleaching alert, a model input) or whether they need an operational product; Part E quotes the error of our fill on hidden pixels next to the trivial baseline and next to NOAA's blended analysis, which is the evidence for that decision.

**Why did we do it?** Naming the audiences up front stops the report from trying to say everything to everyone. The grader needs the rubric answered item by item. An engineer needs the configuration advice and the caveats. A learner needs the explanations in Parts B and D of this handbook. Keeping the four apart also makes it obvious where the project is still incomplete, which is a decision in itself: whether to submit with laptop-only measurements or to run the cluster jobs first.

**How does it work?** The student's decision is what to run before 13 November 2026 and how to present it. The project is 35 percent of the module grade. The evidence in hand covers four of the five rubric items with laptop data. The fifth, scaling to at least 256 cores, has job scripts written and generated (a single-node job plus one job per multi-node rank count under `jobs/`) but not yet submitted, and the graphics-card test needed when OpenMP is the chosen second layer has a working, validated program but no GPU measurement. The decision is therefore to book time on the Atlas and Vanda clusters, run `scripts/atlas/gen_jobs.sh --submit` and `qsub scripts/vanda/gpu.pbs`, and rerun the analysis script on the returned CSV files. Until that happens the report must say, and does say, which figures are provisional.

The grader's decision is how many of 35 points to award. The rubric from lecture 12 is five points for explaining the algorithm (Sections 2.1 to 2.4 of the report), five for the serial runtime against problem size (Section 5 and the serial scaling figure), ten for the MPI analysis with strong and weak scaling, speedup and efficiency, and the impact of communication (Section 6, with separate timers for halo exchange and global sums), ten for the MPI+X analysis including a GPU test (Section 7 and the hybrid and OpenMP figures. The GPU part is where evidence is missing), and five for the discussion (Section 8). The grader can also check the validation table in Section 4, which was not asked for in points but was required in the deliverables.

The engineer's decisions are the ones the design experiments answer. How many MPI ranks per node and how many threads each? On this laptop eight ranks of two threads won three of the four tests and tied the fourth. The lecture 9 question of 1 rank with 72 threads versus 2 with 36 versus 4 with 18 on a Vanda node is the same question and the cluster hybrid jobs will answer it. Which exchange strategy? Non-blocking messages are the safe default. Overlapping computation with communication helped in 2D (10 percent faster) and hurt in 3D (22 percent slower). One-sided puts through a fence should be avoided with this MPI on this platform. How often to check convergence? Every ten iterations costs nothing measurable. Every iteration costs 2.6 times the run. What shape of cut? Keep the faces that are exchanged contiguous in memory. In 3D, slabs cut along the contiguous direction took 0.42 seconds where slabs cut the other way took 0.30 with the same volume of data. What is still missing for the engineer is anything about a real interconnect: every message here crossed shared memory inside one laptop.

The learner's decision is which Part of this handbook to read. Part B teaches the methods from nothing, Part D explains every file, Part E explains every figure. The learner is also the reason every section answers the same five questions in the same order.

**What came out of it?** A clear division of labour between the report and this handbook, and a short, explicit list of unfinished work: the 256-core runs, the GPU run, and a cluster-scale repeat of the hybrid split. Nothing else is pending.

**How would I explain it to a classmate in two sentences?** The report is written for the grader against the rubric, and this handbook is written for someone learning the subject. The laptop results are complete and the cluster results are scripted but not yet run. If you only want to know how to configure a stencil code, read Section 6.3 to 6.5 and Section 7.2 of the report and keep the laptop caveat in mind.

## A4. The storyline of the report chapter by chapter

**What is it?** The report in `report/report.pdf` has ten numbered sections plus references. Its spine is the sequence "what problem, how we parallelised it, is it right, how fast alone, how fast together, how fast with the second layer, what it means". Each section rests on a specific figure or table produced by `analysis/summarize.cpp` and `analysis/plots.gp` from the CSV files in `results/local/`.

**Why did we do it?** The order follows the rubric and the way a sceptical reader thinks. You cannot believe a speedup until you believe the answer is correct, so validation comes before performance. You cannot judge a parallel time until you know the serial time it is compared against, so serial scaling comes before MPI scaling. The second layer (threads or GPU) is a refinement of MPI, so it comes last among the results.

**How does it work?** Section 1, the introduction, states the problem in three paragraphs (the satellite holes, the equation, the three parallel forms), says why the problem is not embarrassingly parallel, and carries a boxed status note saying which measurements are from the laptop and which await the cluster. Section 2 is the algorithm in six parts: the equation and its grid form (2.1), the Jacobi sweep and why it needs about four times the square of the grid size in sweeps (2.2), conjugate gradient with its two global sums per iteration (2.3), the manufactured test solution and the story of the exponential factor that had to be added after the first version let conjugate gradient finish in one step (2.4), a cost model that counts operations and bytes per point and messages per iteration (2.5), and the application (2.6): where the data come from, how a cloud hole becomes a Laplace problem with the clear pixels as fixed values and the coast as an insulated wall, and how the fill is tested by hiding known pixels. Section 3 is the parallel design: the Cartesian cut of the grid (3.1), the masked stencil that skips land and fixed pixels on the satellite data (3.2), the four halo exchange strategies (3.3), the global reductions and the residual check (3.4), the OpenMP layer and its pragmas (3.5), the CUDA layer and the emulation used to validate it (3.6), and the "one source, five executables" arrangement that makes the comparisons fair (3.7).

Section 4, validation, rests on three tables: the discretisation error in 2D and 3D with the measured ratio per doubling (3.88 rising to 3.99, order 2.00), the agreement table in which twenty-seven runs through every implementation report the same error to ten printed digits, and the Jacobi-to-tolerance table showing serial and parallel stopping at the same sweep (15700 in 2D, 2350 in 3D). Section 5, serial scaling, rests on the serial scaling figure with its two panels (total time and time per iteration against grid points, both on log-log axes) and the table of update rates, about 1.6 to 2.3 billion Jacobi updates per second and about 0.4 billion conjugate gradient updates per second on one core.

Section 6, MPI performance, is the longest. Its strong scaling figure (three panels against core count on log axes: time, then speedup, then efficiency) and table carry the headline 8.34 times speedup at 12 cores for conjugate gradient in 2D and the collapse at 16 cores. The communication-share figure shows the halo and all-reduce time rising from under 2 percent at 2 cores to 68 to 82 percent at 16. The weak scaling figure holds the work per rank constant and shows the time staying flat to 4 cores and rising after. The chart of exchange modes compares the four strategies on a log axis because the one-sided variant is off the scale. The chart of process-grid shapes compares cuts, and the check interval figure shows the cost of the global residual check. Section 7, MPI+X, has the hybrid-split figure (16x1 down to 1x16 ranks by threads at a fixed 16 cores) and the OpenMP-only thread scaling figure, and a sub-section stating that the GPU figure will be produced by `scripts/vanda/gpu.pbs` and is absent.

Section 8 is the satellite data: the maps of one cloudy day as observed, filled in 2D and filled in 3D (8.1), the accuracy of the fill on hidden pixels against the size of the hidden patches and against the weight given to the time neighbours (8.2), and the strong scaling of the solver on the real field with its land and ragged holes (8.3). Section 9, the discussion, draws the threads together: which part of the problem needs the most communication (the all-reduce in conjugate gradient at high rank counts, the halo wait in Jacobi), why Jacobi is bound by memory bandwidth, why the laptop's mixed cores cap useful parallelism at 12, what the data say about filling clouds in space against space-and-time, what the cluster runs should show, and what would be done next (a preconditioner, deeper halos, pipelined conjugate gradient, smarter time weights). Section 10 states how AI assistance was used. Section 11 gives the exact commands to download the data and regenerate everything, and the references list the lecture slides, the data sources and the standard texts.

**What came out of it?** A report whose every number can be traced to a CSV row and whose every figure can be regenerated with one command. The status box in Section 1 and the explicit sub-sections for pending cluster figures mean a reader is never misled about what has been measured.

**How would I explain it to a classmate in two sentences?** The report goes problem, design, correctness, serial speed, MPI speed, hybrid speed, discussion, in that order because each step earns the right to the next. Sections 4 to 7 are each built on one or two figures from the analysis script, and Section 1 tells you plainly which figures still await the cluster.

## A5. How to use this handbook

**What is it?** This handbook is a companion to the report, written for someone with no background who wants to understand every number and every file. It has six Parts. Part A (this one) is the overview. Part B teaches the methods from scratch: the equation, finite differences, Jacobi, conjugate gradient, domain decomposition, performance measurement, threads and GPUs. Part C is about the data: the satellite dataset (what it measures, where it comes from, what is in it and how it is stored), the synthetic test problem, and then every input number, every output column, every assumption and its sensitivity. Part D walks the folder file by file, quoting the actual code. Part E explains every figure and table in the report. Part F has the glossary, thirty likely examiner questions with model answers, a ten-minute presentation script keyed to the report sections, and a troubleshooting guide for tracing any number back to its input.

**Why did we do it?** A report has to be short and aimed at a grader. The thinking behind it, the mistakes made along the way, and the meaning of each line of code do not fit there, yet they are what a newcomer needs. Writing them down also forced an honest audit: every claim in the handbook had to be checked against a file, and where a file did not exist (the cluster results, a real GPU run) the handbook says so rather than guessing.

**How does it work?** Every section in every Part answers the same five questions in the same order, in bold at the start of each paragraph: what is it, why did we do it, how does it work, what came out of it, and how would I explain it to a classmate in two sentences. If you are in a hurry, read only the first and last of the five in each section. If you want depth, the middle three carry the mechanism and the evidence. Technical terms are defined the first time they appear within a Part, so each Part can be read on its own. The glossary in Part F collects every definition alphabetically.

To find the code behind a number, start in Part E, which names the figure or table and the function in `analysis/summarize.cpp` that produced it, then follow the pointer to the CSV column and to the solver function in Part D that wrote that column. Part F's troubleshooting section gives the same route as a checklist, from a suspicious number in a figure back through `tables.md`, the CSV row, the command line in `run_all.sh`, and the C++ function that computed it.

**What came out of it?** A document that is long because it is complete, not because it repeats itself. Each Part was written against the actual files in the project folder, and three places are marked as unfinished work rather than smoothed over: the 256-core runs, the GPU run, and the cluster-scale hybrid split.

**How would I explain it to a classmate in two sentences?** Read Part A for the story, Part B to learn the methods, Part D to understand the code, and Part E to read the figures. Part F is for the exam. Every section asks the same five questions, so once you know the pattern you can skim or dig as you like.
# Part B. The methods, taught from scratch

This Part explains every method the project uses, starting from nothing. Each section first says what the thing is, then why we used it, then how it works, then what it produced, and finally how you could explain it to a classmate in two sentences. Where a number helps, the number comes from this project's own files and runs. Where a timing result is quoted, it comes from the figures and tables explained in Part E.

One template note before we start. The brief for this handbook asks each method to state its "functional unit", "system boundary", "criteria", "objective function" and "constraints". Those words come from life-cycle assessment and decision analysis, and our project has no such model. The nearest honest equivalents, used throughout this Part, are these. The functional unit is one solve of the Poisson problem on a grid with n points per side to a stated tolerance or sweep count. The system boundary is the iteration loop only (setting up the grid, writing files and the final error check are timed separately or not at all). The criteria are wall-clock time, speedup, efficiency, the share of time spent communicating, and the error against the exact answer. The objective is the lowest wall-clock time at a given accuracy, and the constraints are memory per process, the 256-core ceiling set by the grading rubric, and the 13 November 2026 deadline.

## B1. The Poisson equation: what problem we are solving

**What is it?** The Poisson equation is a rule that says how a smooth quantity spread over a region must look once everything has settled down, given what is pushing on it from inside and what is pinned at the edges. Written in symbols it is $-\nabla^2 u = f$, read "minus the Laplacian of u equals f". The Laplacian of a function at a point is a measure of how much the function's value there differs from the average of its immediate surroundings: positive when the point is a dip below its neighbours, negative when it is a bump above them. The function $u$ is the unknown we want (a temperature, a pressure, a height), and $f$ is a known source (a heater, a weight) that we are told in advance. An everyday picture is a stretched soap film on a bent wire frame. The wire is the boundary and fixes the film's height at the edge. If nobody pushes on the film, each point settles at exactly the average height of its neighbours, which is Laplace's equation, the special case $f = 0$. If you blow gently on one spot, the film bulges there, and the amount of bulge at each point is set by the Poisson equation with $f$ describing your breath.

In this project the region is the unit square (two dimensions, $0 < x, y < 1$) or the unit cube (three dimensions, with $z$ as well). The boundary condition is $u = 0$ on every edge or face, which engineers call a homogeneous Dirichlet condition. In the soap-film picture the wire frame is flat and at height zero all the way round, so every bulge is caused by $f$ alone. The lecturer's project notes say that anyone solving the Poisson equation must do both the 2D and the 3D version, which is why every binary in this project takes a `--dim 2` or `--dim 3` flag.

**Why did we do it?** We needed an algorithm that is simple enough to write correctly in two weeks but whose parallel version is not trivial. The Poisson equation is the standard such problem in high-performance computing. It appears inside weather models (the pressure step), in gravity and electrostatics, in image editing (pasting one photo into another without a visible seam solves a Poisson problem on the pixels), and as the hardest step in most fluid simulations. More importantly for the grading rubric, the natural way to parallelise it forces neighbouring processes to exchange data on every iteration, so communication cost is real and measurable rather than an afterthought. Lecture 12 of the course uses exactly this pattern, called a stencil with halo exchange, as its worked example of domain decomposition.

**How does it work?** We cannot solve the equation with pen and paper for an arbitrary $f$, so we test the code with a trick called a manufactured solution: pick the answer first, then work out which $f$ would produce it. If the code is right it must reproduce the answer we chose. The chosen answer, written in `src/common.hpp`, is

```cpp
inline double exact_at(const Params& p, const Local& L, long gi, long gj, long gk) {
    const double x = gi * L.hx;
    double v = std::exp(x) * std::sin(PI * x) * std::sin(PI * gj * L.hy);
    if (p.dim == 3) v *= std::sin(PI * gk * L.hz);
    return v;
}
```

that is $u = e^{x} \sin(\pi x)\sin(\pi y)$ in 2D and the same times $\sin(\pi z)$ in 3D. The sine factors are zero at $x = 0$, $x = 1$, $y = 0$ and $y = 1$, so $u$ obeys the boundary condition automatically. Differentiating twice and changing sign gives the matching source, which the code computes in `rhs_at`:

```cpp
inline double rhs_at(const Params& p, const Local& L, long gi, long gj, long gk) {
    const double x = gi * L.hx;
    double v = std::exp(x) * std::sin(PI * gj * L.hy);
    if (p.dim == 3) v *= std::sin(PI * gk * L.hz);
    return v * ((p.dim * PI * PI - 1.0) * std::sin(PI * x) - 2.0 * PI * std::cos(PI * x));
}
```

The factor $e^{x}$ deserves an explanation because it was not the first choice. The first version of the code used the plain product $\sin(\pi x)\sin(\pi y)$, which is the textbook example. It passed every Jacobi test, but conjugate gradient (Section B4) converged in exactly one iteration on every grid, which was suspicious. The reason is that a product of sines is an eigenvector of the discrete operator: applying the stencil to it just rescales it by a constant (about 18.75 on the coarse 3x3 example grid of Section B2, close to the continuous value $2\pi^2 = 19.74$). When the right-hand side is a multiple of an eigenvector, conjugate gradient finds the exact answer in one step, so the benchmark would have measured nothing. Multiplying by $e^{x}$ breaks the symmetry, spreads the solution across many eigenvectors, and makes CG take the realistic number of iterations reported below. The mistake cost an afternoon and is the kind of thing worth saying out loud in a report.

**What came out of it?** A problem that every variant of the code (serial, OpenMP, MPI, MPI+OpenMP, emulated MPI+CUDA) solves to the same answer, with an exact solution available at every grid point so the error can be measured rather than guessed. Section B2 shows the error numbers.

**How would I explain it to a classmate in two sentences?** The Poisson equation says each point of a settled surface sits at the average of its neighbours, nudged by whatever is pushing on it from inside. We solve it on a square and a cube with the edges pinned at zero. We chose the answer in advance (an exponential times sines) so we can check the computer against it, and we made sure the answer is not a special case that would make the harder solver look better than it is.

## B2. Finite differences: turning a smooth problem into a grid of numbers

**What is it?** A finite-difference method replaces a smooth function, which has a value at infinitely many points, by its values at a finite grid of points, and replaces derivatives by differences between neighbouring values. Think of a photograph: the real scene is continuous, but the camera stores a grid of pixels, and you can still tell where edges are by comparing a pixel with the one next to it. A grid point is one pixel, the grid spacing $h$ is the pixel size, and a derivative becomes "the difference between this pixel and the next, divided by $h$".

**Why did we do it?** Computers store numbers, not functions. Finite differences are the simplest way to make the Poisson equation computable, and they produce exactly the neighbour-averaging structure that makes the parallel version interesting: every grid point needs only its immediate neighbours, so a process holding one block of the grid needs only a one-pixel border from the processes next door.

**How does it work?** Take $n$ interior points per dimension. With the two boundary points at 0 and 1 that makes $n + 2$ points and $n + 1$ gaps, so the spacing is $h = 1/(n+1)$ and interior point $i$ sits at $x = ih$ for $i = 1, \dots, n$. `make_local` in `src/common.hpp` sets exactly this:

```cpp
    L.hx = 1.0 / (gn[0] + 1); L.hy = 1.0 / (gn[1] + 1);
    L.ihx2 = 1.0 / (L.hx * L.hx); L.ihy2 = 1.0 / (L.hy * L.hy);
    if (p.dim == 3) { L.hz = 1.0 / (gn[2] + 1); L.ihz2 = 1.0 / (L.hz * L.hz); }
    else            { L.hz = 1.0; L.ihz2 = 0.0; }
    L.diag = 2.0 * (L.ihx2 + L.ihy2 + L.ihz2);
```

The names are worth decoding once. `ihx2` is "inverse h x squared", that is $1/h_x^2$. `diag` is the coefficient that multiplies the centre point, $2/h_x^2 + 2/h_y^2 + 2/h_z^2$. In 2D the code sets `ihz2` to zero, so the z terms vanish and the same formulas serve both dimensions. The second derivative in $x$ at a point is approximated by (left neighbour minus twice the centre plus right neighbour) divided by $h_x^2$. Adding the $y$ (and $z$) versions and changing sign gives the discrete equation at every interior point:

$$\text{diag}\cdot u_c - \frac{u_l + u_r}{h_x^2} - \frac{u_d + u_u}{h_y^2} \left[- \frac{u_b + u_f}{h_z^2}\right] = f_c ,$$

where $l, r, d, u, b, f$ stand for the left, right, down, up, back and front neighbours. The pattern of a centre with four (2D) or six (3D) neighbours is called a 5-point or 7-point stencil, a stencil being the fixed shape you stamp on every grid point. The code applies it in `apply_A` in `src/kernels.hpp`:

```cpp
                Ap[c] = diag * p[c] - ihx2 * (p[c - 1] + p[c + 1])
                                    - ihy2 * (p[c - sy] + p[c + sy])
                                    - ihz2 * (p[c - sz] + p[c + sz]);
```

Here the grid is stored as one long array, `c` is the position of the centre point in that array, and `sy` and `sz` are the strides, the number of array slots you jump to move one step in $y$ or $z$ (moving one step in $x$ is one slot). The function is called `apply_A` because the whole set of equations is a matrix equation $A u = f$, and this line computes one entry of $A$ times a vector without ever storing the matrix. That matrix-free approach is what makes the memory footprint just a few arrays the size of the grid.

A worked example makes the sizes concrete. Take 2D with $n = 3$, so $h = 1/4$, $1/h^2 = 16$ and diag $= 64$. The nine interior points sit at $x, y \in \{0.25, 0.5, 0.75\}$. The exact solution $e^{x}\sin(\pi x)\sin(\pi y)$ at those points, row by row from $y = 0.25$ to $0.75$ and left to right, is

| | x = 0.25 | x = 0.50 | x = 0.75 |
|---|---|---|---|
| y = 0.25 | 0.64201 | 1.16582 | 1.05850 |
| y = 0.50 | 0.90794 | 1.64872 | 1.49695 |
| y = 0.75 | 0.64201 | 1.16582 | 1.05850 |

and the source $f$ at the centre point is 30.896. Solving the nine linear equations exactly (by Gaussian elimination, done by hand for this example, nothing from the project code) gives 1.71586 at the centre against the true 1.64872, an error of 0.067. The largest error over the nine points is 0.072. That is a big error, because $h = 1/4$ is a very coarse grid: four pixels across cannot capture a curve. If you instead apply the stencil to the exact solution values at the centre you get 29.734 rather than the true 30.896, and that gap, the truncation error, is the whole reason the discrete answer differs from the smooth one.

The payoff is how fast the error shrinks as the grid is refined. The scheme is second-order accurate, meaning the error is proportional to $h^2$: halve the spacing and the error falls by four. The project's validation runs confirm this. Running the serial conjugate-gradient solver to a tolerance of $10^{-10}$ in 2D gave an L2 error (a root-mean-square error weighted by cell area) of $5.266\times10^{-4}$ at $n = 32$, $1.357\times10^{-4}$ at $n = 64$ and $3.445\times10^{-5}$ at $n = 128$. The ratios are $5.266/1.357 = 3.88$ and $1.357/0.3445 = 3.94$, both close to 4. The exponent, $\log(3.94)/\log(2) = 1.98$, is the measured order of accuracy. (The grid doubles from 33 to 65 gaps, not exactly from 32 to 64, which is why the ratio is a little under 4.)

**What came out of it?** A grid of $n^2$ or $n^3$ unknowns per solve, a stencil with one line of arithmetic per point, and a verified second-order error. The error numbers above are in `results/local/validation.csv` under the tags `order2d` and `order3d`, and Part E tabulates them.

**How would I explain it to a classmate in two sentences?** We chop the square into a grid of pixels and replace "average of the surroundings" with "the four or six pixels next door", which turns the equation into one simple arithmetic rule per pixel. Doubling the pixel count per side cuts the error by four, and we checked that on the real code.

## B3. Jacobi iteration

**What is it?** Jacobi iteration is the simplest way to solve the grid equations: start with any guess, and repeatedly replace every point by the value the stencil says it should have given its neighbours' current values. It is like a room full of people each asked to set their own temperature to the average of their neighbours'. Nobody coordinates, everyone looks at last round's values, and after many rounds the room settles. The "settled" state is exactly the solution of the equations, because a state that no longer changes satisfies every stencil equation at once.

**Why did we do it?** Jacobi is the clearest possible first solver, it is bitwise reproducible across any number of processes (which makes it a perfect correctness check, see below), and it is a classic stencil benchmark: a fixed number of Jacobi sweeps is what most papers use to measure how fast a machine can run a halo-exchange code. It is also slow to converge, which is why Section B4 adds conjugate gradient.

**How does it work?** Rearranging the stencil equation from Section B2 to isolate the centre value gives the update rule. In `src/kernels.hpp` it is one line inside three nested loops:

```cpp
                un[c] = (f[c] + ihx2 * (u[c - 1] + u[c + 1])
                              + ihy2 * (u[c - sy] + u[c + sy])
                              + ihz2 * (u[c - sz] + u[c + sz])) * inv_diag;
```

Every new value `un[c]` is computed from the old array `u` and the source `f`, then multiplied by `inv_diag`, which is $1/\text{diag}$ precomputed once so the inner loop multiplies instead of dividing (division is several times slower on a CPU). Two arrays are needed because every point must read its neighbours' old values. If we overwrote in place, a point's left neighbour would already hold a new value while its right neighbour still held an old one, and the method would become a different scheme (Gauss-Seidel) that is harder to parallelise. After each sweep the solver swaps the two pointers, so no copying happens:

```cpp
            std::swap(cur, nxt);
```

On the coarse $n = 3$ example of Section B2, one sweep from a zero guess sets the centre to $f_c / \text{diag} = 30.896/64 = 0.483$, against the final value 1.716. The next sweep uses that 0.483 for the neighbours, and so on.

How many sweeps are needed? Each sweep damps the error by a factor at best $\cos(\pi h)$, which for small $h$ is about $1 - \pi^2 h^2/2$, a number very close to 1 when the grid is fine. To shrink the error by a factor $10^{-10}$ you therefore need roughly $\ln(10^{-10}) / \ln(\cos \pi h)$ sweeps, which grows like $n^2$. The project's own runs match this formula closely:

| n | h | predicted sweeps | measured sweeps | ratio to previous |
|---|---|---|---|---|
| 32 | 1/33 | 5074 | 5050 | |
| 64 | 1/65 | 19706 | 19650 | 3.89 |
| 128 | 1/129 | 77639 | 77300 | 3.93 |

Doubling $n$ multiplies the sweep count by almost 4 and the work per sweep by 4 (in 2D), so total Jacobi time grows like $n^4$. At $n = 1024$ the formula predicts about 5 million sweeps, each touching a million points, which is why the timing experiments run Jacobi for a fixed 200 sweeps as a stencil benchmark rather than to convergence, and leave the "solve to tolerance" job to conjugate gradient.

The stopping rule when a tolerance is given is the relative residual. The residual is $r = f - Au$, the amount by which the current guess fails to satisfy the equations, and the code stops when its size relative to the size of $f$ falls below `--tol`:

```cpp
                rel = std::sqrt(C.sum(local)) / fnorm;
                if (P.tol > 0 && rel < P.tol) { converged = 1; break; }
```

`C.sum` is a global sum across all processes (one process's share of the residual is not enough. The test must be on the whole grid), and that global sum is an MPI all-reduce, the communication step in which every process contributes a number and every process receives the total. Computing the residual also requires a fresh halo exchange and a full extra pass over the grid. For that reason the check is not done every sweep. The option `--check-every C` sets how often. `solve` in `src/solver.hpp` defaults it to 10 when a tolerance is given and otherwise never checks, and the experiments vary it between 1, 10, 100 and never to measure what the check costs.

**What came out of it?** Jacobi with a fixed sweep count gave bit-for-bit identical answers from every implementation. For 300 sweeps on a 300x300 grid the relative residual was 0.9707, the L2 error 0.8370888 and the maximum error 1.703428 whether the run used the serial binary, 4 OpenMP threads, 6 MPI ranks in each of the four exchange modes, MPI with threads, or the emulated CUDA driver. The residual is still large because 300 sweeps is far short of the roughly 340,000 the formula predicts for a tolerance of $10^{-8}$ at $n = 300$. The point of the test is agreement, not accuracy. The validation script also runs Jacobi to a tolerance of $10^{-8}$ on a 64x64 grid in serial and on 4 ranks (tag `jacobi_conv` in `results/local/validation.csv`). Because the arithmetic is identical and the check happens every 50 sweeps, both must stop at the same sweep, and Part E reports the recorded counts.
Measured on the laptop (Part E5 and E11): one Jacobi sweep over a 2048 by 2048 grid takes 2.56 ms on one core, which is 1.6 to 2.3 billion point updates per second across all grid sizes, and checking the residual after every sweep at 16 ranks raised the runtime from 0.216 s to 0.566 s, while checking every 10 sweeps cost nothing measurable.

**How would I explain it to a classmate in two sentences?** Jacobi repeatedly sets every grid point to what the stencil says it should be given its neighbours' last values, and stops when the equations are satisfied to a chosen tolerance. It is dead simple and perfectly parallel inside each sweep, but it needs about four times as many sweeps every time you double the grid resolution, so we use it as a benchmark and as a correctness check rather than as the production solver.

## B4. Conjugate gradient

**What is it?** Conjugate gradient (CG) is a smarter iterative solver for the same grid equations. Where Jacobi takes many small, uncoordinated steps, CG treats the problem as finding the lowest point of a bowl-shaped surface and takes a sequence of carefully chosen straight-line steps, each in a direction that never undoes the progress made by earlier steps. The usual picture is a hiker descending into a long narrow valley in fog. A naive hiker always walks straight downhill, which in a narrow valley means zig-zagging from wall to wall and making slow progress along the valley floor. The CG hiker remembers the direction of the last step and adjusts the next one so it is "conjugate" to it, a technical word meaning that moving along the new direction does not spoil the minimisation already achieved along the old one. In exact arithmetic the CG hiker reaches the bottom of a bowl with $N$ dimensions in at most $N$ steps, and in practice far fewer.

**Why did we do it?** Jacobi needs about $4n^2$ sweeps to converge, which is hopeless for large grids. CG needs a number of iterations proportional to $n$, so it is the solver you would actually use. It also brings a second kind of communication into the project: besides the halo exchange that both solvers need, CG performs two global sums per iteration, so the balance between neighbour-to-neighbour messages and all-to-all reductions can be measured and discussed. The grading rubric specifically asks for the impact of communication on runtime, and having two solvers with different communication patterns gives that discussion substance.

**How does it work?** CG works with four vectors the size of the grid: the current guess $u$, the residual $r = f - Au$, a search direction $p$, and the product $Ap$. The loop in `src/solver.hpp`, with the timing lines removed for readability, is:

```cpp
        std::copy(f.begin(), f.end(), r.begin());      // u = 0  =>  r = f
        std::copy(f.begin(), f.end(), p.begin());      // first direction is the residual
        double rr = C.sum(dot(L, r.data(), r.data()));
        for (it = 1; it <= P.iters; ++it) {
            exchange_and_apply(C, L, p.data(),
                [&](Range rg) { applyA(p.data(), Ap.data(), rg); }, t_compute);
            const double pAp = C.sum(dot(L, p.data(), Ap.data()));
            const double alpha = rr / pAp;
            axpy(L,  alpha, p.data(),  u.data());      // u = u + alpha p
            axpy(L, -alpha, Ap.data(), r.data());      // r = r - alpha A p
            const double rr_new = C.sum(dot(L, r.data(), r.data()));
            rel = std::sqrt(rr_new) / fnorm;
            if (P.tol > 0 && rel < P.tol) { converged = 1; break; }
            xpay(L, r.data(), rr_new / rr, p.data());  // p = r + beta p, beta = rr_new / rr
            rr = rr_new;
        }
```

Read it step by step. `rr` is the squared length of the residual, the quantity being driven to zero. `apply_A` computes $Ap$ using the stencil, and because $p$'s neighbours on other processes are needed, `exchange_and_apply` refreshes the halo first. `alpha` is the step length: how far to walk along $p$ to reach the lowest point of the bowl along that line. `axpy` is the standard name for "y plus alpha times x". The two calls move the guess and update the residual without recomputing $Au$ from scratch. `rr_new` measures the new residual. If it is small enough we stop. Otherwise `beta` says how much of the old direction to blend into the new one, and `xpay` ("x plus beta times y", our name for the $p$ update) builds the next direction. The two `C.sum` calls inside the loop are the two global sums per iteration mentioned above. Each is an MPI all-reduce on the parallel builds and a no-op on the serial one.

A two-equation example shows the arithmetic exactly. Take $A = \begin{pmatrix}4 & 1\\ 1 & 3\end{pmatrix}$ and $f = (1, 2)$, so the answer is $u = (1/11, 7/11) = (0.0909, 0.6364)$. Start with $u = 0$, so $r = p = (1, 2)$ and $rr = 5$. Iteration 1: $Ap = (6, 7)$, $pAp = 20$, $\alpha = 5/20 = 0.25$, so $u = (0.25, 0.5)$ and $r = (1, 2) - 0.25\,(6, 7) = (-0.5, 0.25)$. $rr_{new} = 0.3125$, $\beta = 0.3125/5 = 0.0625$, and the new direction is $p = (-0.5, 0.25) + 0.0625\,(1, 2) = (-0.4375, 0.375)$. Iteration 2: $Ap = (-1.375, 0.6875)$, $pAp = 0.859375$, $\alpha = 0.3125/0.859375 = 0.3636$, so $u = (0.0909, 0.6364)$ and $r = (0, 0)$ exactly. Two unknowns, two iterations, done, as the theory promises.

On the real grids the iteration count grows in proportion to $n$. The validation runs to tolerance $10^{-10}$ gave 93 iterations at $n = 64$, 187 at 128, 371 at 256, 738 at 512 and 1467 at 1024 in 2D: each doubling of $n$ almost exactly doubles the count (ratios 2.01, 1.98, 1.99, 1.99). In 3D the counts were 22, 53, 113 and 228 for $n = 16, 32, 64, 128$. Compare Jacobi's 77,300 sweeps at $n = 128$ with CG's 187 iterations: even though one CG iteration costs roughly two to three Jacobi sweeps' worth of arithmetic, CG is over a hundred times cheaper on that grid and the gap widens with $n$.

The limitation is that we use plain CG with no preconditioner. A preconditioner is a cheap approximate inverse applied at each step that reshapes the bowl to be rounder so fewer steps are needed. Good ones (multigrid in particular) make the iteration count independent of $n$. We left that out to keep the communication pattern simple and the code short, and the report says so.

**What came out of it?** A solver that reaches a relative residual of $10^{-10}$ in about $1.4n$ iterations in 2D, with every parallel variant agreeing with the serial one. For $n = 300$ in 2D and tolerance $10^{-9}$, serial CG took 432 iterations and reached L2 error $6.327714\times10^{-6}$. The MPI, MPI+OpenMP and emulated CUDA runs took the same 432 iterations and reported the same error to all seven printed digits, with the residual differing only in its third significant figure because the global sums are added in a different order on different process counts. In 3D at $n = 45$ the count was 76 and the error $2.061051\times10^{-4}$ on every variant.
Measured (Part E5 and E6): CG to tolerance 1e-8 took 3.75 s on the 1024 by 1024 grid and 30.9 s on 2048 by 2048 on one core (1458 and 2909 iterations), and on 12 cores the 1024 grid took 0.46 s, a speedup of 8.34 at 70 percent efficiency.

**How would I explain it to a classmate in two sentences?** Conjugate gradient solves the same equations as Jacobi but picks each step so it never undoes earlier steps, which cuts the iteration count from about $4n^2$ to about $1.4n$. The price is two global sums per iteration, which on a cluster means every process has to stop and agree on a number twice per step.

## B5. Domain decomposition and halo exchange

**What is it?** Domain decomposition means cutting the grid into blocks and giving each block to a different process, so the blocks can be updated at the same time on different processors. A process in this context is an independent copy of the program with its own memory. In the MPI library each one is called a rank and is numbered from 0. Because every stencil update needs the four or six neighbours, a point on the edge of a block needs a value that lives in the next block, held by another rank. The fix is to give each block a one-point-wide border of extra storage, the halo (also called ghost cells), and to refresh that border with the neighbour's edge values before every sweep. The refresh is the halo exchange. Picture a jigsaw cut into 16 rectangles handed to 16 people who must each repaint their piece so that every pixel matches the average of its neighbours. Before each round, each person photocopies the outer line of pixels of their piece and hands the copies to the people holding the adjacent pieces.

**Why did we do it?** It is the natural, and in practice the only sensible, way to parallelise a stencil across machines that do not share memory. Each rank's work is proportional to the volume of its block, while its communication is proportional to the surface, so bigger blocks mean relatively less talking. Lecture 12 presents this surface-to-volume reasoning and the halo exchange as the central pattern for parallel grid codes, and the rubric rewards an explanation of which parts of the problem communicate most.

**How does it work?** Three steps: build a grid of ranks, give each rank its block, and exchange halos.

The grid of ranks is built in `make_decomp` in `src/decomp.hpp`:

```cpp
    MPI_Dims_create(D.size, 3, D.dims);            // fills in the zeros as evenly as possible
    int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, D.dims, periods, 1, &D.cart);
    MPI_Comm_rank(D.cart, &D.rank);
    MPI_Cart_coords(D.cart, D.rank, 3, D.coords);
    for (int d = 0; d < 3; ++d) MPI_Cart_shift(D.cart, d, 1, &D.lo[d], &D.hi[d]);
```

`MPI_Dims_create` takes the total number of ranks and proposes a process grid as close to a cube as possible: 16 ranks become 4x4x1 for a 2D problem (the third entry is forced to 1) and 4x2x2 for a 3D one. 256 ranks become 16x16x1 or 8x8x4. The user can override this with `--dims`. `MPI_Cart_create` then arranges the ranks in that grid so that each one can ask who its neighbours are. `MPI_Cart_shift` answers, returning the special value `MPI_PROC_NULL` on the outer boundary, which later makes sends and receives to the "missing" neighbour silently do nothing. That is how the physical boundary condition $u = 0$ gets in: the halo on the outside of the whole grid is never written and keeps its initial zeros.

Each rank's block is cut out by `partition` in `src/common.hpp`:

```cpp
inline void partition(long n, int P, int coord, int& count, long& offset) {
    long base = n / P, rem = n % P;
    count = (int)(base + (coord < rem ? 1 : 0));
    offset = coord * base + (coord < rem ? coord : rem);
}
```

For $n = 10$ points over $P = 4$ ranks, `base` is 2 and `rem` is 2, so the first two ranks get 3 points and the last two get 2, at offsets 0, 3, 6 and 8. Every point is owned by exactly one rank and the counts differ by at most one, which keeps the work balanced. For the laptop experiments, 2048 points over a 4x4 grid give each rank 128 points in each direction with no remainder, and 192 points over 16 slabs give 12 each.

The halo exchange is where the design choices live. First, how to describe the data. The left and right faces of a block are not contiguous in memory (the grid is stored row by row), so instead of copying them into a temporary buffer, `face_type` in `src/decomp.hpp` builds an MPI derived datatype that describes the face in place:

```cpp
inline MPI_Datatype face_type(int nx, int ny, int nz, int d, int pos) {
    int sizes[3]    = {nz + 2, ny + 2, nx + 2};  // C order: slowest first (k, j, i)
    int subsizes[3] = {nz, ny, nx};
    int starts[3]   = {1, 1, 1};
    int cd = 2 - d;                               // dimension d=0 (x) is the last C index
    subsizes[cd] = 1; starts[cd] = pos;
    MPI_Datatype t;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &t);
    MPI_Type_commit(&t);
    return t;
}
```

`MPI_Type_create_subarray` says: inside an array of `sizes` (the block plus halos), take the sub-box of `subsizes` starting at `starts`. Setting one dimension's sub-size to 1 and its start to `pos` picks a single plane: `pos = 1` is the first interior plane (what we send to the lower neighbour), `pos = 0` is the lower halo (where we receive from it), `pos = n_d` is the last interior plane and `pos = n_d + 1` the upper halo. The MPI library then gathers the scattered values itself, often faster than a hand-written copy loop. Eight such types per rank describe every face and every halo.

Second, how to move the data. Four strategies are implemented so their cost can be compared, selected with `--exchange`:

`sendrecv` uses the blocking call `MPI_Sendrecv`, one direction at a time: send my upper face to the upper neighbour while receiving the lower neighbour's upper face into my lower halo, then the reverse. Blocking means the call does not return until the data has gone, so the three directions happen one after another.

`isend` posts every receive and every send at once with `MPI_Irecv` and `MPI_Isend` (the I stands for immediate: the call returns straight away and the transfer proceeds in the background), then waits for all of them with `MPI_Waitall`. All directions overlap with each other, and the network has the whole set of messages to schedule.

`overlap` posts the messages like `isend`, but instead of waiting immediately it computes the deep interior of the block, the points that are at least two away from any halo and therefore never read a halo value, and only then waits and finishes the one-point-thick outer layer. `exchange_and_apply` in `src/solver.hpp` does this using `deep_interior` and `for_each_boundary_slab` from `src/kernels.hpp`. If the network is slow and the block is large, the messages are already there by the time the interior is done, and the communication is hidden.

`rma` uses one-sided communication. Every rank exposes its array as an MPI window, and a neighbour writes its face directly into my halo with `MPI_Put`, without me posting a matching receive. A pair of `MPI_Win_fence` calls marks the start and end of the exchange. Because the target rank's block may be one point bigger or smaller than mine, `make_decomp` recomputes the neighbour's block size and builds a second set of datatypes, `nbr_lo_halo` and `nbr_hi_halo`, that describe where in the neighbour's array the data must land.

Third, how much data moves. The surface-to-volume ratio decides it. On the 2048x2048 grid split over 16 ranks as 4x4 blocks, each rank owns 512x512 = 262,144 points and sends at most four faces of 512 values, 2048 values per sweep, or 0.8 percent of what it computes. Split instead as 16 slabs of 128x2048, each rank still owns 262,144 points but sends two faces of 2048, 4096 values, twice as much. In 3D on 192x192x192 over 16 ranks, the 4x2x2 grid gives blocks of 48x96x96 with a surface of 36,864 values against a volume of 442,368, about 8 percent, while 16x1x1 slabs of 12x192x192 send 73,728 values, double again. Three-dimensional problems have proportionally more surface, so they communicate more for the same number of points, which is one reason the 3D efficiency numbers in Part E are lower than the 2D ones. The shape also affects contiguity: a face perpendicular to $x$ is strided in memory (expensive to gather) while a face perpendicular to $z$ is one contiguous chunk, so the cheapest shape is not always the one with the smallest surface. The experiment behind `results/local/decomp.csv` varies `--dims` to measure exactly this trade-off.

**What came out of it?** A decomposition that gives identical answers for every process grid and every exchange mode (Section B3 lists the agreement numbers), eight derived datatypes per rank, and a switchable exchange strategy. The lecturer's rubric item on communication is answered by timing `t_halo` (time spent in the exchange) and `t_allreduce` (time in global sums) separately from `t_compute`, which the CSV output records as both the maximum and the mean over ranks.
Measured (Part E7, E9 and E10): at 12 ranks communication took 16 percent of Jacobi 2D's time and 47 percent of Jacobi 3D's. Non-blocking exchange was the best default. Overlap saved 6 to 10 percent in 2D but cost 16 to 22 percent in 3D. Fence-based RMA was 2.5 to 7 times slower, and in 3D, slabs with strided faces (16x1x1, 0.420 s) were 40 percent slower than slabs with contiguous faces (1x1x16, 0.295 s) for the same number of bytes.

**How would I explain it to a classmate in two sentences?** We cut the grid into blocks, one per process, and give each block a one-pixel border that is refreshed from the neighbours before every sweep. Communication grows with the surface of a block while work grows with its volume, so squarish blocks on big grids talk little, and we built four different ways of doing the refresh to see which is fastest.

## B6. Measuring parallel performance

**What is it?** Parallel performance is measured by comparing how long a job takes on one processor with how long it takes on many. Speedup is the serial time divided by the parallel time, $S(p) = T_1 / T_p$ for $p$ processors. Efficiency is speedup divided by $p$, $E(p) = T_1 / (p\,T_p)$, the fraction of the extra processors that actually bought you something. If 16 people paint a wall that one person paints in 16 hours, perfect teamwork finishes in 1 hour (speedup 16, efficiency 1). If they get in each other's way and take 2 hours, the speedup is 8 and the efficiency 0.5, meaning half the crew's time was wasted on waiting and coordination.

**Why did we do it?** The rubric gives 10 of 35 marks for a scaling analysis to at least 256 cores with both strong and weak scaling and with speedup and efficiency reported, and 5 more for the serial runtime against problem size. These quantities are also the only honest way to compare the design choices in Section B5: a halo-exchange strategy that saves 5 percent at 16 ranks might save 30 percent at 256.

**How does it work?** There are two ways to grow a parallel experiment, and they answer different questions. Strong scaling holds the problem fixed and adds processors: "I have this grid, how much faster can I get it done?" Each rank's block shrinks as $p$ grows, so its surface-to-volume ratio worsens and communication eventually dominates. Weak scaling grows the problem in step with the processors so that each rank always has the same amount of work: "I have twice the machine, can I solve a problem twice as big in the same time?" The ideal weak-scaling result is a flat line. The code supports weak scaling directly through `--local-n M`, which sets the global grid to M points per rank in each direction times the process grid, so with 16 ranks as 4x4 and `--local-n 1024` the global grid is 4096x4096.

Two laws from Lecture 7 bound what is possible. Amdahl's law says that if a fraction $S$ of the run cannot be parallelised and the rest, $P = 1 - S$, splits perfectly, the speedup on $n$ processors is $1/(S + P/n)$, which can never exceed $1/S$ however many processors you add. With $S = 0.05$ the speedup is 9.1 on 16 processors and 18.6 on 256, against a ceiling of 20. With $S = 0.01$ it is 13.9 and 72.1. In our solver the serial fraction is tiny (every grid point is updated in parallel) but communication plays the same role: time spent waiting for messages is time not spent computing, and it grows as the blocks shrink. Gustafson's law is the weak-scaling counterpart: if the parallel part grows with $n$ while the serial part stays fixed, the scaled speedup is $S + P\,n$, so with $S = 0.05$ and 256 processors it is 243, almost the full 256. That is why weak scaling looks so much kinder than strong scaling and why the rubric asks for both.

Four practical rules shape how the numbers are collected. First, the serial baseline $T_1$ is the genuinely serial binary `bin/poisson_serial`, not the MPI binary run on one rank. The latter carries MPI overhead and would flatter the speedup. The experiment scripts record both so the overhead itself can be seen. Second, plots with time, problem size or processor count on an axis use logarithmic scales, as the lecturer asked, because those quantities span factors of a hundred or more and because a power law such as "time grows like $N^1$" becomes a straight line whose slope is the exponent. The `slope()` function in `analysis/summarize.cpp` fits those slopes and records them in `summary.txt`. Third, every configuration is run at least twice and the minimum time is kept. Anything that disturbs a run (another program waking up, a page of memory being fetched from disk) can only make it slower, never faster, so the minimum is the best estimate of the intrinsic cost. The mean would mix in the noise. Fourth, the timed region is the iteration loop alone, between two barriers that make all ranks start together. Grid setup and the final error check are outside it, because they are the same for every variant and would dilute the differences.

The communication share is reported as $(t_{halo} + t_{allreduce}) / t_{total}$ using the maximum over ranks of each timer. The maximum, not the mean, matters because the slowest rank sets the pace: every sweep ends with an exchange that cannot complete until the slowest neighbour arrives.

**What came out of it?** A set of CSV files (`results/local/serial_scaling.csv`, `strong.csv`, `weak.csv` and the design-study files) with one row per run, and a script that turns them into the speedup, efficiency and communication-share figures in Part E. The laptop runs stop at 16 cores because that is what the machine has. The same scripts, through `scripts/atlas/gen_jobs.sh`, generate the jobs for 1 to 256 cores on the university cluster, and those cluster results had not yet been produced when this handbook was written.
Measured (Part E5, E6 and E8): serial time per iteration grew with a log-log slope of 1.00 to 1.05 in all four cases. Strong-scaling efficiency at 12 ranks was 0.70 (CG 2D), 0.55 (CG 3D), 0.40 (Jacobi 2D) and 0.28 (Jacobi 3D). Weak efficiency at 8 ranks was 0.76 and 0.83 for CG and 0.49 and 0.58 for Jacobi. At 16 ranks every case lost a factor of about two to the four efficiency cores.

**How would I explain it to a classmate in two sentences?** Speedup is how many times faster the parallel run is than the serial one, and efficiency is that number divided by the processor count, so 1.0 means no waste. We measure it two ways, keeping the problem fixed (strong scaling) and growing it with the machine (weak scaling), always against a true serial baseline and always taking the fastest of repeated runs.

## B7. MPI+OpenMP: threads inside a rank

**What is it?** OpenMP is a way to split the work of a loop across the cores of one computer using threads. A thread is a sequence of instructions that runs on one core. Several threads of the same program share the same memory, so they can all read and write the same grid array without sending messages. MPI ranks, by contrast, have separate memories and must exchange messages to see each other's data. The hybrid model MPI+OpenMP uses MPI between computers (or between groups of cores) and OpenMP threads within each rank. Think of a company with several offices: MPI is the courier service between offices, OpenMP is the colleagues in one office who can simply look at the same whiteboard.

**Why did we do it?** The project title is "MPI+X", and OpenMP is the X the course spends Lecture 9 on. Hybrid parallelism also answers a real question the lecturer poses in that lecture: a Vanda node has 72 cores, so should a job run 72 MPI ranks per node, or 1 rank with 72 threads, or something between? Fewer ranks mean bigger blocks, fewer and larger halo messages and fewer participants in each all-reduce, but threads bring their own overhead and the memory bandwidth of a node is shared however it is split. The only way to know is to measure.

**How does it work?** OpenMP is driven by compiler directives, lines beginning `#pragma omp` that the compiler turns into threaded code and that a compiler without OpenMP support simply ignores. That last property is used deliberately: `src/kernels.hpp` is compiled twice, once without the OpenMP flag to make the serial and pure-MPI binaries and once with it to make `bin/poisson_omp` and `bin/poisson_mpi_omp`, so there is exactly one copy of the arithmetic and no risk of the variants drifting apart. The directive on the Jacobi sweep is

```cpp
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = r.k0; k < r.k1; ++k)
        for (int j = r.j0; j < r.j1; ++j) {
            const size_t row = L.idx(0, j, k);
#pragma omp simd
            for (int i = r.i0; i < r.i1; ++i) {
```

`parallel for` says to split the iterations of the following loop across threads. `collapse(2)` says to treat the two outer loops over $k$ and $j$ as one long loop before splitting, which matters in 2D where the $k$ loop has only one iteration and would otherwise give all the work to one thread. `schedule(static)` divides the iterations into equal contiguous chunks decided in advance, the cheapest option when every iteration costs the same, as it does here. The inner `#pragma omp simd` is not about threads at all: it asks the compiler to use the core's vector instructions so that several consecutive $i$ values are updated in one machine instruction. For the dot products and the residual the directive gains a `reduction(+:sum)` clause,

```cpp
#pragma omp parallel for collapse(2) schedule(static) reduction(+:sum)
```

which gives each thread its own private `sum`, lets them accumulate independently, and adds the partial sums together at the end. Without it, every thread would write to the same variable at the same time and the result would be wrong, a race condition.

The interaction with MPI is kept deliberately simple. `src/mpi_main.cpp` initialises MPI with

```cpp
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
```

where `MPI_THREAD_FUNNELED` promises that only the main thread will ever call MPI. All the pragmas end before any exchange or all-reduce, so the pattern in each iteration is: threads compute, threads join, the main thread communicates, threads fan out again. This forgoes the chance to have one thread communicate while the others compute, but it avoids the hardest class of bugs, and the overlap mode of Section B5 recovers some of the same benefit by other means. The thread count comes from the environment variable `OMP_NUM_THREADS` or from `--threads T`, and the binary records it in the CSV so each row knows its ranks-times-threads total.

The experiment behind `results/local/hybrid.csv` keeps the total core count fixed at 16 and tries every split: 16 ranks with 1 thread each, 8x2, 4x4, 2x8 and 1x16. A companion experiment in `omp_threads.csv` runs the OpenMP-only binary on 1 to 16 threads so that shared-memory scaling can be compared with MPI scaling on the same machine. On the cluster, `gen_jobs.sh` writes the same sweep for 256 cores across four nodes, from 256x1 down to 4x64.

**What came out of it?** The hybrid binary agrees with the serial one in every validation case (Section B3 and B4 list the numbers, which were produced with 2 and 4 threads per rank). The timing comparison between splits is a measured result, not something one can reason out in advance, because it depends on the laptop's memory system and on the MPI library's behaviour when ranks share a node.
Measured (Part E12 and E13): eight ranks with two threads each was fastest, or within 6 percent of fastest, for all four problems. Sixteen single-threaded ranks were 1.5 to 1.9 times slower and one rank with sixteen threads was 10 to 25 percent slower. Pure OpenMP reached a speedup of 5.1 on 16 threads for Jacobi 2D relative to one thread (4.1 relative to the serial binary).

**How would I explain it to a classmate in two sentences?** OpenMP lets the cores inside one computer share the work of each loop through shared memory, while MPI carries data between computers, so the hybrid program uses both at once. The same kernel source compiles into every variant, and we measure how to split a fixed number of cores between ranks and threads instead of guessing.

## B8. MPI+CUDA: a GPU per rank

**What is it?** A GPU (graphics processing unit) is a processor built from thousands of small cores that all execute the same instruction on different data at the same time. CUDA is NVIDIA's language for writing programs for its GPUs. A GPU function is called a kernel and is launched over a grid of thread blocks. Each block holds a few hundred threads that run together on one part of the chip, and each thread works out from its block and thread indices which piece of data it is responsible for. If a CPU core is a skilled chef who can cook any dish, a GPU is a thousand line cooks who all chop the same vegetable at once: slower at any one decision, vastly faster at a huge pile of identical work. A stencil sweep, where a million grid points all get the same five-line update, is exactly that pile. In the MPI+CUDA model each MPI rank owns one GPU and keeps its block of the grid in the GPU's memory.

**Why did we do it?** The rubric says that if OpenMP is the chosen X, at least one GPU test is required as well, and Lecture 10 covers CUDA. Beyond the rubric, a GPU port exposes a different communication problem: the halo now lives in GPU memory, and getting it to the network means either copying through the CPU or asking the MPI library to read GPU memory directly, which is a measurable design choice.

**How does it work?** The CUDA driver `src/cuda/mpi_cuda_main.cu` reuses the decomposition from Section B5 unchanged and replaces the arithmetic with GPU kernels in `src/cuda/gpu_kernels.cuh`. The Jacobi kernel is

```cpp
__global__ void k_jacobi(Geo g, Box b, const double* u, double* un, const double* f) {
    int i = b.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    int j = b.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    int k = b.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= b.i1 || j >= b.j1 || k >= b.k1) return;
    size_t c = cell(g, i, j, k);
    un[c] = (f[c] + g.ihx2 * (u[c - 1] + u[c + 1]) + g.ihy2 * (u[c - g.sy] + u[c + g.sy])
                  + g.ihz2 * (u[c - g.sz] + u[c + g.sz])) / g.diag;
}
```

The arithmetic is the same single line as the CPU version. The difference is who executes it: instead of a loop, every thread computes its own $(i, j, k)$ from the built-in `blockIdx`, `blockDim` and `threadIdx` values and updates one point, and threads that fall outside the box return without doing anything. The block shape is set by `box_block()`, which returns `dim3(64, 4, 1)`: 64 threads along $x$, 4 along $y$, 1 along $z$, 256 threads per block. Sixty-four along $x$ is chosen because $x$ is the contiguous direction in memory, so the 64 threads of a row read 64 consecutive doubles. The GPU memory system serves such consecutive reads in a few wide transactions, a property called coalescing. Threads reading scattered addresses would each need their own transaction and run many times slower. `box_grid` divides the box size by the block size, rounding up, to get the number of blocks.

Dot products need a different pattern because a million threads cannot all add into one number at once. `k_dot` uses a grid-stride loop: a fixed launch of `REDUCE_BLOCKS = 1024` blocks of `REDUCE_THREADS = 256` threads, 262,144 threads in all, each of which steps through the array in strides of 262,144, sums its share privately, and then adds that one partial sum to the result with `atomicAdd`. An atomic add is an addition that the hardware guarantees will not be interrupted by another thread's addition to the same address, so 262,144 of them produce the correct total. This is not the fastest possible reduction (a shared-memory tree would do fewer atomics), but it is short, correct, and has no synchronisation inside a block, which also lets the emulation layer described below run it.

The halo exchange has two paths. In the default path, a `k_pack` kernel copies each face of the block into a contiguous GPU buffer, `cudaMemcpy` moves that buffer to pinned (page-locked) host memory, the usual `MPI_Isend` and `MPI_Irecv` carry it to the neighbour, and the reverse copy and a `k_unpack` kernel put the received face into the halo. With the `--cuda-aware` flag the host copies are skipped and MPI is given the GPU buffer addresses directly, which works only when the MPI library on the machine was built with CUDA support (Lecture 10 calls this CUDA-aware MPI). The `overlap` mode works as on the CPU: after the sends are posted, the deep-interior kernel is launched asynchronously and runs while the messages are in flight, then the driver waits for the messages and unpacks them, and only then updates the outer layer.

Now the honest part. This laptop has no NVIDIA GPU and no `nvcc` compiler, so the CUDA code has not been compiled by CUDA or run on a GPU. What has been done is this: `src/cuda/cuda_compat.hpp` provides, when `nvcc` is absent, a stand-in for the CUDA runtime in which `cudaMalloc` is `malloc`, `cudaMemcpy` is `memcpy`, `atomicAdd` is a plain addition, and a kernel launch becomes a sequential loop over every block and thread index that calls the kernel function once per thread with the index globals set. The binary `bin/poisson_mpi_cuda_emul` built this way runs the real driver logic, the real pack and unpack kernels and the real MPI calls, just slowly and on the CPU. It reproduces the serial results to every printed digit in 2D and 3D, for Jacobi and CG, in `isend` and `overlap` modes, and with and without `--cuda-aware`. That proves the indexing, the halo bookkeeping and the solver logic are right. It does not prove that the code compiles under `nvcc` (there may be syntax the real compiler rejects), that kernel launches are configured legally for a real device, or anything about speed. The job script `scripts/vanda/gpu.pbs` contains the compile and run steps for Vanda, and the GPU results in Part E will be empty until it has been run.

**What came out of it?** A complete MPI+CUDA driver and kernel set, validated for correctness through CPU emulation on 3 to 8 ranks, plus a job script and a Makefile target (`make cuda CUDA_ARCH=sm_80`) for the real GPU build. No GPU timing exists yet.
Not yet measured: the GPU figure (`gpu_vs_cpu.png`) and the multi-GPU table are produced by `make figures` from `results/cluster/gpu.csv` and `gpu_multi.csv`, which exist only after `scripts/vanda/gpu.pbs` and `gpu_multi.pbs` have run on Vanda.

**How would I explain it to a classmate in two sentences?** A GPU runs the same one-line stencil update on thousands of grid points at once, so each MPI rank hands its block to a GPU and only the one-pixel borders travel over the network, through the CPU or directly if the MPI library can read GPU memory. We verified the GPU code's logic by running it on the CPU through an emulation layer, but it has not yet been compiled or timed on a real GPU.
## B9. Filling a hole in a map with the same equation

**What is it?** The way a cloud hole in a satellite temperature map is turned into the Poisson problem of B1, and why the answer is the "smoothest possible" fill. Everything in B1 to B8 carries over; this section only explains what changes.

**Why did we do it?** Parts B1 to B8 solve a problem whose answer is known in advance. The satellite maps are the problem whose answer is unknown, and it is worth seeing, in the smallest possible example, why "every missing pixel is the average of its neighbours" is the right rule and what it does and does not know.

**How does it work?** Start in one dimension with a row of seven pixels along a line of sea. The first two are observed at 28 and 28, the last two at 30 and 31, and the three in the middle are under cloud. Call the unknowns a, b and c. The rule says each unknown equals the average of its two neighbours: a = (28 + b)/2, b = (a + c)/2, c = (b + 30)/2. Three equations, three unknowns. Solving them gives a = 28.5, b = 29.0, c = 29.5: the fill is a straight line from the last observed value on the left to the first on the right. That is the whole idea. In one dimension the smoothest fill is a straight line; in two dimensions it is the shape a soap film takes when its rim is pinned to the observed pixels around the hole; in three dimensions, with time as the third axis, the film is pinned also to yesterday's and tomorrow's values at the same place.

Now put land on the left: pixels one and two are coast, and a has only one sea neighbour, b. The rule becomes a = b (the average of a's one neighbour), and the solution flattens out towards the coast: nothing is assumed to flow across the shoreline. That is what the code does by dropping land neighbours from the average and from the diagonal (D4). It is the zero-flux or Neumann boundary condition, and it is a choice: the alternative, extending the sea temperature onto the land as a fixed value, would be wrong because there is no sea temperature on land.

The same three equations can be written as a matrix. Put the unknowns on the left and the observed values on the right: 2a - b = 28, -a + 2b - c = 0, -b + 2c = 30. The matrix has 2 on the diagonal and -1 next to it: it is the one-dimensional Laplace matrix of B2 with the source set to zero, and the observed neighbours have moved to the right-hand side. In two and three dimensions the diagonal is 4 or 6 (or less next to land, or different when the time weight is not 1), and the right-hand side at each unknown collects the weights times the values of its observed neighbours. This is the matrix the solver's masked kernels apply without ever writing it down.

Jacobi and conjugate gradient then work exactly as in B3 and B4. Jacobi takes the rule literally, sweeping the hole and replacing each unknown by the average of its neighbours' current values, and needs about as many sweeps as the square of the hole's width in pixels to settle, because information crawls in from the rim one pixel per sweep. A hole 200 pixels wide needs tens of thousands of sweeps, which is why the real runs use conjugate gradient, which needs a few hundred to a few thousand iterations for the whole map. The mean observed temperature is subtracted first and added back at the end, so that the solver works on small numbers and so that a region with no observed rim at all (a bay never seen all year) ends up at the mean rather than at zero degrees.

What the method does not know is also visible in the seven-pixel example. If the true temperature under the cloud had a bump, a warm eddy at b of 31 degrees, the fill would still give 29.0, because nothing in the rim says the bump is there. In two dimensions the fill smooths over any front or eddy that lies entirely under a cloud; in three dimensions it recovers the bump if it was visible yesterday or tomorrow at the same place, which is why the year helps. Part E measures how large that effect is.

**What came out of it?** The maps and tables of E17 to E19: a fill that beats the trivial guess by a wide margin on single hidden pixels and by less as the hidden patches grow, and a comparison of the 2D and 3D fills on the same hidden pixels that shows when the neighbouring days help and how much weight they should get. On the test day, 64 km hidden squares are filled to 0.15 degrees from the day alone; the year-long fill is worse than that for small holes unless the time weight is tiny, and better for 250 km holes.

**How would I explain it to a classmate in two sentences?** A missing pixel is set to the average of its sea neighbours, which makes the fill the smoothest surface pinned to the clear pixels around the hole, like a soap film on a wire; land is left out so the coast acts as a wall. It is the same Poisson solver, with the heater off and the fixed edges moved to wherever the clouds end, and it cannot invent a feature it never saw unless the neighbouring days show it, and even then only if those days are trusted with the right weight.

# Part C. The data

## C1. The two kinds of data: a synthetic test problem and a real satellite dataset

**What is it?** The project works on two kinds of data. The first is synthetic: a problem generated by a formula the moment the program starts, used for validation and for the speed benchmarks. People call data produced this way synthetic data, numbers made by a rule rather than collected from the world. The rule is called a manufactured solution: pick the answer first and then build the question whose answer it is. A maths teacher does the same thing when setting an exam: she decides the answer should be 7, then writes "3x + 1 = 22" so that marking is easy and the correct answer is known with certainty. The second kind is real: 365 daily maps of sea-surface temperature for 2025 over the box 5 N to 25 N and 65 E to 95 E (the Arabian Sea, the Bay of Bengal, India and Sri Lanka), measured by the VIIRS instrument on the NOAA-20 satellite at a pixel size of 0.0375 degrees, about 4 km. Each map is 801 pixels wide and 534 tall, 427 734 pixels, of which 142 547 (33 percent) are land. A sea pixel on a given day is either observed (a temperature in degrees Celsius) or missing, which almost always means cloud. The whole year is 156 million pixels.

**Why did we do it?** The assignment allows any algorithm as long as the parallel version is hard to build, and says a dataset is not required. The manufactured solution gives us a yardstick that no dataset can: because we know the exact answer, we can compute the error of every run to fifteen digits, and because it is a formula we can ask for a grid of any size, which is what weak scaling needs (Lecture 12 made this point). But a project built by data science students for a mathematics course should also answer a real question, and satellite temperature maps come with one. Infrared satellites cannot see through cloud, so their maps have holes, and filling the holes is a daily task in oceanography and climate monitoring (heat waves, coral bleaching, fisheries, weather-model inputs). The simplest filling method, demanding the smoothest field that matches the clear pixels, is exactly the Laplace equation our solver solves. The seas around India were chosen because the monsoon makes them among the cloudiest seas on Earth, so the holes are large and real, and because a single island like Singapore is only a dozen pixels across at this resolution.

**How does it work?** The synthetic formula is written in two short functions in `src/common.hpp`, `exact_at` and `rhs_at`. The chosen answer is u = exp(x) sin(pi x) sin(pi y) in two dimensions, multiplied by sin(pi z) in three. Taking minus the Laplacian (the sum of the second derivatives, the quantity that measures how curved a surface is) of that answer by hand gives the right-hand side f = exp(x) sin(pi y) [sin(pi z)] ((dim pi^2 - 1) sin(pi x) - 2 pi cos(pi x)). The program fills an array with f, pretends it has never seen u, solves for u, and then compares what it found against the formula. The factor exp(x) is there for a reason recorded in the file's comment: a plain product of sines happens to be a special direction (an eigenvector) of the grid operator, which made the conjugate gradient method finish in a single step and made every timing of it meaningless. The exponential breaks that symmetry.

The satellite data come from the NOAA CoastWatch ERDDAP server (https://coastwatch.noaa.gov/erddap). ERDDAP is a data server that lets you ask for a slice of a gridded dataset by writing the latitude, longitude and date range into the web address, and sends back a plain text file. No account is needed. The script `scripts/data/download.sh india 2025 ref` asks for the VIIRS temperature of each day of 2025 in our box (dataset `noaacwN20ACSPOSSTL3GCDaily`, about 1 MB per day), the land mask of NOAA's blended analysis (dataset `noaacwBLENDEDsstDaily`, variable `mask`), and, because of the word `ref`, the blended analysis temperature itself for every day. The blended analysis is a gap-free product that NOAA makes by merging infrared and microwave satellites with ship and buoy measurements through a statistical method called optimal interpolation; we use it as an independent reference, not as the truth. The downloads land in `data/raw/india/` as 731 text files. The tool `bin/gridtool convert` (source in `tools/gridtool.cpp`) then stacks the 365 daily files into one binary file, `data/india_2025.grid`: an 84-byte header (grid size, the longitude and latitude of the first pixel and the spacing, the first date) followed by one 32-bit floating-point number per pixel in the order day, latitude, longitude, with the special value NaN ("not a number") for a missing pixel and -999 for land. `bin/gridtool regrid` makes a second file of the same shape from the blended analysis, `data/india_2025_ref.grid`. The solver reads its own block of these files directly (`src/gridio.hpp`), so the raw text files are needed only once and are not part of the submission.

The pixel is the unit of length and the day is the unit of time, so the grid spacing is 1 in every direction, and the option `--tweight` says how strongly a pixel is tied to the same pixel on the neighbouring days relative to its four neighbours in space (1 means equally).

**What came out of it?** Two sorts of files. From the synthetic problem, every run yields three correctness numbers and nine timing numbers, written as one line of a comma-separated file (a CSV, a plain text table with commas between the columns, the simplest spreadsheet format there is). Nine such files exist in `results/local/`: `validation.csv`, `serial_scaling.csv`, `strong.csv` and `weak.csv`, and `exchange.csv`, `decomp.csv`, `checkevery.csv`, `hybrid.csv` and `omp_threads.csv` from the later stages of the script. From the satellite data, `bin/gridtool info data/india_2025.grid --days` reports how much of the sea was observed on each day: 42.5 percent over the whole year, with monthly means of the missing share of 44 percent in January, 25 in February, 29 in March, 45 in April, 70 in May, 84 in June, 88 in July, 84 in August, 76 in September, 64 in October, 43 in November and 35 in December. The observed temperatures range from 7.3 to 36.9 degrees with a mean of 28.4. The day used for the 2D experiments, 10 April 2025 (index 99), has 65 percent of its sea pixels missing. The runs on the data write `results/local/gapfill.csv` (same columns plus the pixel counts and fill errors described in C3), the filled fields `results/fields/fill2d_day99.grid` and `fill3d.grid`, and the map figure `results/figures/sst_maps.png`. The cluster files under `results/cluster/` do not exist yet because nobody has run the cluster jobs.

**How would I explain it to a classmate in two sentences?** For speed and correctness we invent a problem whose answer we already know, solve it, and check we got it back. For the real question we download a year of daily satellite sea-temperature maps around India from NOAA, stack them into one file, and fill the cloud holes with the same solver, checking it by hiding pixels we know.

## C2. Every number that goes INTO a run

**What is it?** A run is controlled by a dozen command-line options, which are words you type after the program name, like `--n 2048`. Each one sets a field of the `Params` structure in `src/common.hpp` (a structure is a named bundle of variables, like a form with labelled boxes). The defaults are written right there in the structure definition, and the scripts that launch experiments override them.

**Why did we do it?** Putting every knob on the command line means the same executable serves validation, serial timing, strong scaling, weak scaling and the design studies. Nothing needs recompiling between experiments, and every CSV row records the values that produced it, so a result can always be traced back to its inputs.

**How does it work?** The defaults in `src/common.hpp` are these:

```cpp
struct Params {
    int dim = 2;                          // 2 or 3
    long n[3] = {256, 256, 1};            // global interior points per dimension (nz=1 for 2D)
    long local_n = 0;                     // weak scaling: if >0, n_d = local_n * dims_d
    std::string solver = "jacobi";        // jacobi | cg
    std::string exchange = "isend";       // sendrecv | isend | overlap | rma
    int dims[3] = {0, 0, 0};              // process grid (0 = let MPI_Dims_create decide)
    long iters = 1000;                    // maximum iterations
    double tol = 0.0;                     // relative residual tolerance (0 = run exactly `iters` sweeps)
    int check_every = 0;                  // jacobi: compute residual every k iterations (0 = never)
    int threads = 0;                      // OpenMP threads override (0 = OMP_NUM_THREADS)
    std::string csv;                      // append a results row to this file
    std::string tag;                      // free text copied to the CSV row
    bool verbose = false;
    bool cuda_aware = false;              // mpi_cuda: pass device pointers straight to MPI
    // --- satellite gap filling -------------------------------------------------
    std::string data;                     // input grid file (see gridio.hpp); empty = manufactured problem
    std::string out;                      // write the filled field to this grid file
    std::string ref;                      // optional gap-free reference grid (same shape as data) for comparison
    double holdout = 0.0;                 // fraction of observed pixels hidden for validation
    double tweight = 1.0;                 // 3D: weight of the time neighbours relative to the space neighbours
    int day = 0;                          // 2D: which time slice of the file to use (0-based)
    int days = 0;                         // 3D: how many slices to use starting at --day (0 = all)
    unsigned seed = 1;                    // seed of the deterministic holdout selection
    int holdout_block = 1;                // side of the square patches hidden together (1 = single pixels)
    int eval_day = -1;                    // 3D: report the hidden-pixel errors for this day only (-1 = all days)
    bool data_mode() const { return !data.empty(); }
};
```

The second block of fields belongs to the satellite data and is ignored unless `--data` names a grid file. `--data` switches the program from the formula to the file and takes the grid size from the file's header, so `--n` is not used. `--day` picks the day for a 2D run (0 is 1 January 2025, 99 is 10 April) and the first day of a 3D run, and `--days` limits how many days a 3D run stacks (0 means all that remain). `--tweight` is the weight of the two time neighbours relative to the four space neighbours. `--holdout 0.1` hides one tenth of the observed pixels so that the fill can be checked against them, `--holdout-block 16` hides them in 16 by 16 patches (64 km squares, more like real clouds than single pixels), and `--seed` changes which pixels are hidden. `--ref` names the reference grid for the comparison with NOAA's blended analysis, `--out` writes the filled field to a grid file, and `--eval-day K` makes a 3D run report its hidden-pixel error for day K alone, so that the year's fill can be compared with the 2D fill of the same day on the same hidden pixels. The experiments use `--holdout 0.1 --holdout-block 16` throughout, patch sides 1, 4, 16 and 64 in the accuracy study, and time weights 0.01, 0.03, 0.1, 0.3, 1 and 3.

`--dim` chooses the square (2) or the cube (3). `--n` is the number of unknown grid points along each side, not counting the fixed boundary. `--n 2048` in 2D means 2048 x 2048 = 4,194,304 unknowns, and `--n 192` in 3D means 192^3 = 7,077,888. You can give different values per direction as `--n 512,256`. `--local-n` is used only for weak scaling: instead of a global size it fixes how many points each process owns per direction, and the program multiplies by the process grid to get the global size, so that doubling the processes doubles the problem. `--solver` picks Jacobi (the simple averaging rule) or CG (conjugate gradient, the smarter one). `--exchange` picks one of four ways of swapping boundary data between processes. `--dims` forces the shape of the process grid, for example `--dims 16,1` to cut the square into 16 horizontal strips. `--iters` is a cap on the number of sweeps, `--tol` is the stopping accuracy, and when `--tol` is 0 the cap is the exact number of sweeps performed. `--check-every` says how often Jacobi stops to measure how far it is from the answer. 0 means never. `--threads` sets how many threads one process uses when compiled with OpenMP. `--csv` and `--tag` say where to record the result and what to call it. The values actually used are set in `scripts/local/run_all.sh` for the laptop and in `scripts/atlas/gen_jobs.sh` and `scripts/vanda/gpu.pbs` for the clusters.

| option | meaning | local values (run_all.sh) | cluster values (gen_jobs.sh, gpu.pbs) | where set |
|---|---|---|---|---|
| `--dim` | 2D square or 3D cube | 2 and 3 for every experiment | 2 and 3 | `Params.dim` |
| `--n` | interior points per side | Jacobi 2D: 128 to 4096 (serial), 2048 (parallel); Jacobi 3D: 16 to 256, 192; CG 2D: 128 to 2048, 1024; CG 3D: 32 to 256, 128 | Jacobi 2D 256 to 8192, 8192; Jacobi 3D 32 to 512, 512; CG 2D 256 to 4096, 4096; CG 3D 32 to 256, 256; GPU 1024 to 8192 and 64 to 512 | `Params.n[3]` |
| `--local-n` | points per rank per side (weak scaling) | Jacobi 2D 1024, 3D 64; CG 2D 512, 3D 48 | same four values; GPU 4096 and 256 | `Params.local_n` |
| `--solver` | jacobi or cg | both | both | `Params.solver` |
| `--exchange` | halo exchange method | isend by default; all four in the exchange study | isend; all four in design_64 and design_256 | `Params.exchange` |
| `--dims` | process grid shape | automatic; forced in the decomposition study (16x1, 8x2, 4x4, 2x8, 1x16; six 3D shapes) | automatic; forced at 256 ranks (256x1, 64x4, 16x16, 4x64, 1x256; six 3D shapes) | `Params.dims[3]` |
| `--iters` | sweep cap | 200 for Jacobi benchmarks; 300 for fixed-iteration CG weak scaling; 1,000,000 cap when a tolerance is set | identical | `Params.iters` |
| `--tol` | stopping accuracy | 0 (fixed sweeps) for Jacobi; 1e-8 for CG timing; 1e-10 and 1e-9 for validation | 0 and 1e-8 | `Params.tol` |
| `--check-every` | residual check interval | 0 except the check-interval study: 0, 1, 10, 100 | 0 except design jobs: 0, 1, 10, 100 | `Params.check_every` |
| `--threads` | OpenMP threads per rank | 1 to 16 (hybrid: ranks x threads = 16) | 1 to 64 (ranks x threads = 256); 36 on the GPU node for the CPU comparison | `Params.threads` |
| ranks (`mpirun -np`) | MPI processes | 1, 2, 4, 8, 16 | 1, 2, 4, ..., 256; 64 per node (`PPN=64` in env.sh) | `mpirun` |
| `--data` | satellite grid file | `data/india_2025.grid` (801 x 534 x 365) | same file, copied up | `Params.data` |
| `--day`, `--days` | day for 2D; first day and count for 3D | day 99 (10 April 2025) in 2D; all 365 days in 3D (`DAYS3D` can shorten it) | same | `Params.day`, `Params.days` |
| `--tweight` | weight of the time neighbours | 1; 0.01, 0.03, 0.1, 0.3, 3 in the accuracy study | same | `Params.tweight` |
| `--holdout`, `--holdout-block` | hidden fraction and patch side | 0.1 and 16; patch sides 1, 4, 16, 64 in the accuracy study | same | `Params.holdout`, `Params.holdout_block` |
| `--ref` | reference grid | `data/india_2025_ref.grid` in the accuracy study | same | `Params.ref` |
| `--out` | filled field output | `results/fields/fill2d_day99.grid`, `fill3d.grid` (serial runs) | same | `Params.out` |

The cluster-side knobs that are not program options live in `scripts/atlas/env.sh`: `QUEUE=parallel` and `GPU_QUEUE=gpu` are placeholders the student must replace with the real queue names from `gstat`, `PPN=64` is the number of ranks per node, `CORES_PER_NODE=72` records what Lecture 9 says about Vanda, `WALL=02:00:00` is the two-hour time limit requested per job, and `MODULES` holds the commands that load the compiler and MPI (`source /app1/ebenv; module load OpenMPI` is an Atlas guess that must be checked against `module avail`).

**What came out of it?** The job files that `scripts/atlas/gen_jobs.sh` writes (one single-node job plus one per multi-node rank count), plus `scripts/local/run_all.sh`, define 180 or so distinct configurations in total. Every one of them writes its options into the CSV row it produces, so the inputs are never separated from the outputs. For example the row for the 16-rank Jacobi 2D strong-scaling run in `results/local/strong.csv` begins `mpi,2,2048,2048,1,jacobi,isend,16,1,4x4x1,200,0,...`, which reads: MPI build, two dimensions, 2048 by 2048 by 1 points, Jacobi, non-blocking exchange, 16 ranks, 1 thread each, process grid 4 by 4 by 1, 200 iterations, did not "converge" (because no tolerance was set).

**How would I explain it to a classmate in two sentences?** Every knob is a command-line flag with a default written in `Params`, and the scripts set the flags for each experiment. The flags are copied into the result row, so you can always see what produced a number.

## C3. Every number that comes OUT of a run

**What is it?** Each run appends one line of 41 comma-separated values to the CSV file named by `--csv`. The function that writes it is `write_csv` in `src/common.hpp`, and the header line it writes first is:

```
impl,dim,nx,ny,nz,solver,exchange,ranks,threads,dims,iters,converged,rel_resid,
l2_err,linf_err,t_total,t_setup,t_compute_max,t_halo_max,t_allreduce_max,
t_compute_mean,t_halo_mean,t_allreduce_mean,check_every,tol,hostname,tag,
data,holdout,holdout_block,tweight,n_obs,n_unknown,n_hidden,n_land,mean_obs,
mae_hidden,rmse_baseline,rmse_ref_fill,rmse_ref_obs,eval_day
```

The first 27 columns are the same for both problems. The last 14 are filled only by runs on the satellite data and are zero or empty otherwise. `data` is the grid file; `holdout`, `holdout_block` and `tweight` echo the options; `n_obs`, `n_unknown`, `n_hidden` and `n_land` count the pixels of each kind in the part of the file that was solved; `mean_obs` is the mean observed temperature, around which the fill is computed; `mae_hidden` is the mean absolute error on the hidden pixels; `rmse_baseline` is the root-mean-square error of the trivial fill that sets every hole to `mean_obs`; and `rmse_ref_fill` and `rmse_ref_obs` are the root-mean-square differences between our field and the reference analysis on the filled pixels and on the observed pixels; `eval_day` is -1 unless the errors were restricted to one day with `--eval-day`. For data runs the two error columns of the first block are reused: `l2_err` holds the root-mean-square error on the hidden pixels and `linf_err` the largest single error, so that the agreement tables work for both problems.

**Why did we do it?** The report needs speedups, efficiencies and a breakdown of where the time goes, and the only honest way to get them is to measure them in the program and keep every raw number. Writing one row per run also makes repeats trivial: run the same command twice and you get two rows, which the analysis later reduces to one.

**How does it work?** The first ten columns are inputs echoed back (see C2), plus `impl`, which names the build: `serial`, `omp`, `mpi`, `mpi_omp`, `mpi_cuda` or `mpi_cuda_emul`. The next five describe the answer. `iters` is how many sweeps actually happened. `converged` is 1 if the tolerance was met and 0 otherwise (always 0 when `--tol` is 0, which is normal for the benchmarks, not an error). `rel_resid` is the relative residual, the size of the leftover mismatch f - A u divided by the size of f, computed by `residual` in `src/kernels.hpp` and reduced across processes in `solve` in `src/solver.hpp`. Think of it as how far the current guess is from satisfying the equation, as a fraction: 1e-8 means one part in a hundred million. `l2_err` and `linf_err` compare the answer with the manufactured formula (`error_norms` in `src/kernels.hpp`): the first is a root-mean-square style average over the grid, scaled by the cell volume hx hy hz so that it does not change with grid size for the same function, and the second is the single worst point. Both are printed with 17 significant digits so that two implementations can be compared to the last bit.

The timing columns are all in seconds and come from the wall clock (`MPI_Wtime` in the parallel builds, `std::chrono::steady_clock` in the serial one). `t_total` is the time of the iteration loop only, from a barrier before the first sweep to a barrier after the last, as measured on the slowest process. `t_setup` is the time to fill the right-hand side and compute its norm, recorded separately and excluded from `t_total`. The next six split the loop time into three buckets, each reported twice: `t_compute_*` is time inside the arithmetic kernels, `t_halo_*` is time inside the boundary exchange (including waiting for a neighbour), and `t_allreduce_*` is time inside the global sums. The `_max` version is the largest value any process saw. The `_mean` version is the average over processes. The gap between them is a direct measure of imbalance: if one process spends much longer waiting in the exchange than the average, it is waiting for a slow neighbour. The three buckets do not always add up to exactly `t_total`, because a few bookkeeping operations (the pointer swap, loop control) are not inside any timer. The difference is small. Finally `check_every` and `tol` echo two options, `hostname` is the computer's name and `tag` is the free label (`jac2d`, `cg3d`, `order2d`, `agree` and so on) that the analysis uses to group rows.

A concrete row, the 8-rank Jacobi 2D run from `results/local/strong.csv` (tag `jac2d`, ranks 8, second repeat, which was the faster one):

```
mpi,2,2048,2048,1,jacobi,isend,8,1,4x2x1,200,0,9.984509e-01,0.85135187580423177,
1.7325433190304218,0.105227,0.004162,0.098739,0.015286,0.001078,0.094540,0.010645,
0.000491,0,0,Prateeks-MacBook-Pro.local,jac2d
```

Reading it: 200 sweeps took 0.105227 s. The slowest process computed for 0.098739 s and the average process for 0.094540 s, so compute was well balanced. The slowest process spent 0.015286 s in halo exchange against an average of 0.010645 s. The all-reduce bucket is tiny (0.001 s) because with `check_every = 0` the only global sum is the one that computes the norm of f before the loop and the final diagnostics, which are not timed, so what remains is the two barriers. The relative residual is still 0.998 after 200 sweeps, which is expected: Jacobi on a 2048-point grid needs millions of sweeps to converge, and the benchmark deliberately does not run that long. The L2 error 0.8513518758042 is identical in the first 13 digits to the serial row's 0.8513518758042508, which is the correctness check.

**What came out of it?** Four CSV files with 42 to 48 rows each (the first line of each being the header), all with the same 27 columns. The serial rows have zeros in every halo and all-reduce column because there is nothing to communicate with. Every parallel row carries the extra information needed to say how fast the run was and where the time went.

**How would I explain it to a classmate in two sentences?** Each run writes one CSV line: what you asked for, how good the answer was, and how long the loop took split into compute, boundary exchange and global sums. The max and mean versions of the time buckets tell you whether the processes were waiting for each other.

## C4. Where each data source lives

**What is it?** A map from every number or definition used in the project to the file it comes from. Everything generated by us lives under `poisson-mpi/results/`. Everything we cite from the course lives in the lecture slides and the assignment text in the folder above.

**Why did we do it?** The handbook's rule is that nothing is described from memory. A reader who doubts a number in the report should be able to open the file and find it.

**How does it work?** The table below lists each item. Where an item is a file we produce, the "produced by" column names the script. Where it is a course document, the column names the slide.

| data item | file | column or variable | produced by | year |
|---|---|---|---|---|
| validation runs (convergence order, serial vs parallel agreement, Jacobi to convergence) | `results/local/validation.csv` | all 27 columns; tags `order2d`, `order3d`, `jacobi_conv`, `agree` | `scripts/local/run_all.sh`, stage 1 | 2026 |
| serial time against problem size | `results/local/serial_scaling.csv` | `nx`, `iters`, `t_total`; tags `jac2d`, `jac3d`, `cg2d`, `cg3d` | `run_all.sh`, stage 2 | 2026 |
| strong scaling 1 to 16 ranks | `results/local/strong.csv` | `ranks`, `dims`, `t_total`, `t_halo_max`, `t_allreduce_max` | `run_all.sh`, stage 3 | 2026 |
| weak scaling 1 to 16 ranks | `results/local/weak.csv` | `nx`, `ny`, `nz`, `ranks`, `t_total` | `run_all.sh`, stage 4 | 2026 |
| exchange mode, process-grid shape, check interval, hybrid split, OpenMP threads | `results/local/exchange.csv`, `decomp.csv`, `checkevery.csv`, `hybrid.csv`, `omp_threads.csv` | as above plus `exchange`, `dims`, `check_every`, `threads` | `run_all.sh`, stages 5 to 7 (running at the time of writing) | 2026 |
| console log with wall-clock stamps of each stage | `results/local/run.log` | lines beginning `---` carry the time of day | `run_all.sh` | 2026 |
| every table in Markdown | `results/figures/tables.md` | one table per experiment | `make figures` (`bin/summarize` + gnuplot) | 2026 |
| headline numbers for the report text | `results/figures/summary.txt` | one `key: value` line each, such as `strong_cg2d: max_cores=16 speedup=3.72792 ...` | `make figures` (`bin/summarize`) | 2026 |
| plot data, one file per case | `results/figures/*.dat` | whitespace-separated columns named in the file's second line, e.g. `strong_cg2d.dat`: cores, t_total, speedup, efficiency, comm_frac | `make figures` (`bin/summarize`) | 2026 |
| figures | `results/figures/*.png` | one file per figure, drawn from the `.dat` files | `make figures` (gnuplot, `analysis/plots.gp`) | 2026 |
| cluster results (256 cores, GPU) | `results/cluster/*.csv` | same 41 columns | `jobs/*.pbs` from `scripts/atlas/gen_jobs.sh`; `scripts/vanda/gpu.pbs`, `gpu_multi.pbs` | not yet produced |
| daily satellite sea-surface temperature, 4 km, with cloud gaps | `data/raw/india/sst_2025-MM-DD.asc` (365 files, downloaded, not submitted) | one text grid per day | NOAA CoastWatch ERDDAP, dataset `noaacwN20ACSPOSSTL3GCDaily` (NOAA-20 VIIRS, ACSPO L3 collated), fetched by `scripts/data/download.sh` | 2025 |
| land mask and gap-free reference analysis | `data/raw/india/mask.asc`, `ref_2025-MM-DD.asc` | byte mask; kelvin temperatures | NOAA CoastWatch ERDDAP, dataset `noaacwBLENDEDsstDaily` (geo-polar blended analysis, 5 km) | 2025 |
| the stacked year and the matching reference, as the solver reads them | `data/india_2025.grid`, `data/india_2025_ref.grid` | 84-byte header + float32 [day][lat][lon]; NaN missing, -999 land | `bin/gridtool convert`, `bin/gridtool regrid` | 2025 |
| gap-filling runs: scaling and accuracy | `results/local/gapfill.csv` | all 41 columns; tags `sst2d_cg`, `sst2d_jac`, `sst3d_cg`, `sst3d_jac`, `acc2d`, `acc3d`, `acc3d_day` | `run_all.sh`, stage 8 | 2026 |
| filled temperature fields | `results/fields/fill2d_day99.grid`, `results/fields/fill3d.grid` | same grid format, no missing pixels | the serial data runs in `run_all.sh` (`--out`) | 2026 |
| the maps | `results/figures/sst_maps.png` | observed, 2D fill, 3D fill, difference | `scripts/local/maps.sh` (gridtool slice + `analysis/maps.gp`) | 2026 |
| definitions of speedup, efficiency, Amdahl's law, Gustafson's law, strong and weak scaling | `../Slides_export/Lecture_7.pdf` | slides "Measuring Parallel Speedup", "Amdahl's Law", "Strong and Weak Scaling" | A. Chen, DSA5208 | 2026 |
| grading scheme and lecturer's requirements (2D and 3D, log axes, 256 cores) | `../Slides_export/Lecture_12.pdf` | slides "Project 3", "Grading", "More Comments" | A. Chen, DSA5208 | 2026 |
| cluster names and job instructions | `../Slides_export/Lecture_8.pdf` | section "HPC Resources" | A. Chen, DSA5208 | 2026 |
| node size (72 cores per Vanda CPU node) | `../Slides_export/Lecture_9.pdf` | slide "For your projects" | A. Chen, DSA5208 | 2026 |
| assignment brief and deadline | `../Project 3: Parallelizing an Algorithm with MPI+X.txt` | whole file | course staff | 2026 |

The only external web addresses in the project are the NUS HPC user guide, https://nusit.nus.edu.sg/hpc/ , quoted on the Lecture 8 slides, and the course's Canvas page, whose address is not in any file we have. The student should paste it into the report's reference list. The row for `results/cluster/` is deliberately blank in the last column: those files will exist only after the student runs `scripts/atlas/gen_jobs.sh --submit` and `qsub scripts/vanda/gpu.pbs` on the clusters and copies the folder back.

**What came out of it?** Every number in the report and in Part E of this handbook can be found in one of the first eleven rows of the table. There is exactly one gap, the cluster row, and it is the gap the assignment's 256-core requirement depends on.

**How would I explain it to a classmate in two sentences?** All our measurements are CSV files under results/local, and the analysis script turns them into the figures, tables and spreadsheet under results/figures. The definitions come from the Lecture 7 and 12 slides, and the 256-core cluster files have not been produced yet.

## C5. The machine the local numbers came from

**What is it?** Every local measurement was made on a single Apple MacBook Pro with an M4 Max processor. It has 16 processor cores, 128 GB of memory, and runs macOS (the kernel reports itself as Darwin 25.6.0). The compilers and libraries were Apple clang 21.0.0 as the C++ compiler, Open MPI 5.0.11 installed through Homebrew as the MPI library, Homebrew's libomp for OpenMP, and gnuplot 6.0 for the figures. No Python is used in this project. The CSV rows record the machine as `Prateeks-MacBook-Pro.local` in the `hostname` column.

**Why did we do it?** It was the computer available while writing and debugging the code, and 16 cores are enough to show the shape of every scaling curve before spending cluster time. The assignment, however, asks for runtime figures from Atlas or Vanda and for scaling to at least 256 cores, so the laptop numbers are a rehearsal, not the final performance.

**How does it work?** The 16 cores are not identical. Twelve are performance cores, built for speed, and four are efficiency cores, built to save battery and noticeably slower. The operating system decides which program runs on which core and the program cannot pin itself to a core on macOS. With up to 12 processes everything can sit on fast cores. With 16 processes, four of them necessarily land on slow cores, and in a halo-exchange algorithm every process waits at every sweep for its neighbours to deliver their boundary, so the whole computation moves at the pace of the slowest process. Sixteen people passing buckets along a chain, four of them children, move at the children's speed, and the adults spend most of their time holding an empty bucket.

**What came out of it?** The effect is plain in `results/local/strong.csv`. For the Jacobi 2D problem (tag `jac2d`, 2048 by 2048 points, 200 sweeps) the best of two runs took 0.5272 s on one core, 0.1052 s on 8 ranks and 0.3023 s on 16 ranks. Eight ranks give a speedup of 0.5272 / 0.1052 = 5.0, an efficiency of 5.0 / 8 = 0.63. Sixteen ranks give a speedup of 0.5272 / 0.3023 = 1.74 and an efficiency of 0.11, so adding the last eight cores made the run almost three times slower. The timing buckets say why: at 8 ranks the slowest process spent 0.0153 s in halo exchange against 0.0987 s computing. At 16 ranks it spent 0.2392 s in exchange against 0.0928 s computing. The compute time per process barely changed (the work per process halved, but the slow cores are about half as fast), while the waiting time rose fifteen-fold. The same pattern appears in every 16-rank row of `strong.csv` and `weak.csv`: in the CG 2D row with 16 ranks, 0.826 s of the 1.229 s total is spent inside `MPI_Allreduce`, which is not slow network hardware, it is fast processes waiting at the meeting point for slow ones. For this reason the 16-rank laptop points should be read as "what heterogeneous cores do to a synchronous algorithm", and the 8-rank points as the laptop's honest parallel performance.

**How would I explain it to a classmate in two sentences?** The laptop has 12 fast cores and 4 slow ones, so runs with 16 processes crawl at the speed of the slow four. Up to 8 processes the numbers are clean. The cluster runs are what the report's 256-core claims must rest on.

## C6. Every assumption we made, why, its plausible range, and how much the answer changes if it changes

**What is it?** Thirteen choices that are baked into the experiments. None of them is wrong, but each could have been made differently, and a reader should know which of our numbers would move if it were.

**Why did we do it?** A performance study is only as good as its stated conditions. The grading scheme rewards an explanation of how design choices affect scaling and efficiency, and that explanation has to start by naming the choices.

**How does it work?** Each assumption below gets its own sub-section with the same five questions. "Plausible range" is the set of values a reasonable person might have chosen instead. "Effect" is our best statement of how the headline numbers (speedup, efficiency, communication share, error) would change, with a measurement where we have one and a marked gap where we do not.

**What came out of it?** Three of the thirteen (the fixed sweep count, the ranks-per-node choice and the residual-check setting) change the parallel numbers materially. The rest change them little or not at all, and the error numbers depend only on the grid, the precision and the boundary condition.

**How would I explain it to a classmate in two sentences?** We list every setting we could have chosen differently and say what it would do to the results. Most would barely move the numbers. The sweep count, the ranks-per-node and the residual-check frequency are the ones that matter.

### Assumption (a): Jacobi is benchmarked for a fixed 200 sweeps, not run to convergence

**What is it?** Every Jacobi timing row uses `--iters 200 --tol 0`, so the program performs exactly 200 sweeps and stops without checking how close it is to the answer. The `rel_resid` in those rows is therefore large (0.998 for the 2048-point grid), and `converged` is 0.

**Why did we do it?** Jacobi needs a number of sweeps proportional to n squared: the validation rows in `validation.csv` show 15,700 sweeps for a 64 by 64 grid to reach 1e-8, which extrapolates to roughly 16 million sweeps for 2048 by 2048. Nobody runs Jacobi that way in practice. It is used here as the classic stencil benchmark, where what matters is the cost per sweep and how that cost splits between arithmetic and communication. Two hundred sweeps is long enough for the per-sweep cost to dominate the one-off costs and short enough that the whole suite runs in minutes.

**How does it work?** The timing columns scale in direct proportion to the number of sweeps as long as nothing else changes, so `t_total / iters` is the quantity to compare across rows, and the analysis script does this for the serial study. The plausible range is 50 to 1000 sweeps.

**What came out of it?** With 50 sweeps the fixed overheads (the two barriers, the first-touch of memory pages, MPI connection setup) would be a larger share and the apparent efficiency a few percent lower. With 1000 it would be a little higher. The speedup and efficiency numbers are otherwise unaffected. The error columns for these rows are meaningless as accuracy measures and should not be read as such.

**How would I explain it to a classmate in two sentences?** Jacobi would take millions of sweeps to converge on a big grid, so we time 200 sweeps as a benchmark. The time per sweep is what we compare, and it would not change with a different count.

### Assumption (b): conjugate gradient stops at a relative residual of 1e-8

**What is it?** The CG timing rows use `--tol 1e-8`, so the method stops when the leftover mismatch is one hundred-millionth of the size of f.

**Why did we do it?** It is tight enough that the error against the exact solution is dominated by the grid, not by the solver (compare `l2_err` 5.4567e-07 in the 1e-8 run on the 1024-point grid in `strong.csv` with 5.4567e-07 in the 1e-10 run on the same grid in `validation.csv`: identical to five digits), and loose enough that the iteration counts stay in the hundreds or low thousands.

**How does it work?** CG's iteration count grows roughly in proportion to log(1 / tol), so the range 1e-6 to 1e-12 changes the count by about plus or minus 40 percent. Because the parallel version does exactly the same iterations as the serial version (the counts in `strong.csv` are 1458 for every rank count on the 2048 grid, and 222 on the 3D grid), speedup and efficiency are unaffected by the tolerance. Only absolute times move.

**What came out of it?** No effect on the comparisons. A proportional effect on the times. A looser tolerance would also show a slightly larger `l2_err`.

**How would I explain it to a classmate in two sentences?** We stop CG when the equation is satisfied to one part in a hundred million. Changing that only changes how long every run takes, not how the runs compare.

### Assumption (c): each configuration is run twice and the faster run is kept

**What is it?** `REPEATS=2` in `scripts/local/run_all.sh`, and `best()` in `analysis/summarize.cpp` keeps the row with the smallest `t_total` for each distinct configuration.

**Why did we do it?** Timing noise on a shared computer only ever adds time (another program woke up, a memory page had to be mapped, a fan curve changed the clock). The minimum of several runs is therefore the best estimate of the intrinsic cost. Two repeats were a compromise with the total running time of the suite.

**How does it work?** Where the two repeats agree closely, the choice is irrelevant: the two serial CG 2D runs on 1024 points took 3.7953 s and 3.7482 s. Where they do not, the minimum removes a disturbance: the two serial Jacobi 2D runs on 4096 points took 1.9566 s and 2.4771 s, and the weak-scaling 8-rank Jacobi run took 0.2352 s and 0.2992 s with the halo time at 0.0152 s versus 0.0745 s, which is the signature of one process being descheduled for a moment.

**What came out of it?** The plausible range is 1 to 10 repeats with minimum or median. With a single run, about one row in ten would carry a 20 to 30 percent disturbance. With five repeats the minima would be at most a few percent lower than ours. The cluster scripts also use two repeats. A student with spare core-hours could raise `for r in 1 2` to `for r in 1 2 3`.

**How would I explain it to a classmate in two sentences?** Timings jitter upwards, never downwards, so we run each case twice and keep the faster one. More repeats would shave a few percent off the slowest-looking rows, nothing more.

### Assumption (d): the halo is one layer deep

**What is it?** Each process stores exactly one extra row of neighbours' values around its block, and refreshes it before every sweep.

**Why did we do it?** The stencil only reaches one point in each direction, so one layer is the minimum that makes a sweep correct. It is also the simplest to code and to verify.

**How does it work?** A deeper halo (two or three layers) lets a process do two or three sweeps between exchanges at the cost of a few redundant computations on the extra layers. This is the communication-avoiding trick mentioned in Lecture 12. It halves or thirds the number of messages. With halo width 1 and the 4 by 4 process grid on the 4096 by 4096 weak-scaling grid, each process sends 4 x 1024 = 4096 numbers per sweep, 32 kilobytes, in four messages.

**What came out of it?** On the laptop the exchange cost is mostly waiting for slow cores, not transfer, so a deeper halo would not have helped. On the cluster, where latency per message is the issue at 256 ranks, it could cut the communication share noticeably. The communication share at 256 ranks in the cluster strong-scaling figure, once it exists, will say how much there is to gain. The code has no deeper-halo option, so this remains a stated limitation.

**How would I explain it to a classmate in two sentences?** We swap one boundary row before every sweep. Swapping two rows every two sweeps would send fewer messages, and we have not implemented it.

### Assumption (e): the starting guess is zero

**What is it?** Both solvers start from u = 0 everywhere (`std::vector<double> u(n, 0.0)` in `src/solver.hpp`).

**Why did we do it?** It is the standard choice when nothing is known about the answer, it makes the first residual equal to f, and it makes every run reproducible.

**How does it work?** A better starting guess (for example the solution on a coarser grid) would reduce the iteration count for CG and the residual reached by Jacobi, but it would not change the cost of one sweep or one iteration, which is what the scaling study measures. The plausible range is zero, random, or a coarse-grid guess.

**What came out of it?** No effect on speedup, efficiency or the per-sweep timings. A random guess would also break the bit-for-bit agreement between implementations unless the same seed were used everywhere.

**How would I explain it to a classmate in two sentences?** We always start from all zeros. A cleverer start would need fewer iterations but would not change the cost of each one.

### Assumption (f): the boundary value is zero everywhere (Dirichlet condition)

**What is it?** The outermost ring of the grid is fixed at zero and never updated. In the code this falls out for free: boundary halo cells are never written, and MPI treats a send to a non-existent neighbour (`MPI_PROC_NULL`) as doing nothing.

**Why did we do it?** The manufactured solution vanishes on the boundary, so zero is the correct value, and a zero boundary needs no extra code path.

**How does it work?** A non-zero boundary would be one extra array filled once at setup. A derivative (Neumann) boundary would change the stencil at the edges and the matrix's properties. Neither changes the amount of communication or arithmetic per sweep.

**What came out of it?** No effect on timings. The error columns would of course change with a different problem.

**How would I explain it to a classmate in two sentences?** The edges are held at zero, which matches our chosen answer and costs nothing. Other boundary conditions would be more code but the same speed.

### Assumption (g): blocks differ in size by at most one point

**What is it?** `partition` in `src/common.hpp` splits n points over P processes so that the first n mod P processes get one extra point. For the 3D strong-scaling grid of 192 points on a 4 by 2 by 2 process grid the blocks are 48 by 96 by 96. For 45 points over 7 processes (the validation run with `dims 7x1x1`) they are 7, 7, 7, 6, 6, 6, 6.

**Why did we do it?** Equal work per process is the goal of any static decomposition, and a difference of one layer is the best possible for a regular grid.

**How does it work?** The imbalance is at most one layer out of the block's thickness: one part in 48 for the 3D strong case, which is 2 percent. With unlucky sizes (say 65 points over 64 processes) one process would have twice the work of the others. The plausible range is this scheme, a scheme that lets block sizes differ by more to align with cache lines, or a dynamic (work-stealing) scheme, which does not fit a stencil.

**What came out of it?** All the sizes used in the experiments divide evenly or nearly so, so the measured `t_compute_max` and `t_compute_mean` are within a few percent of each other wherever the cores are homogeneous: 0.0987 versus 0.0945 for the 8-rank Jacobi 2D row. The large gaps at 16 ranks come from core speed, not block size.

**How would I explain it to a classmate in two sentences?** The grid is shared out as evenly as integer division allows, with at most one extra row for some processes. For the sizes we chose that imbalance is a percent or two.

### Assumption (h): 64 ranks per node on the cluster, not 72

**What is it?** `PPN=64` in `scripts/atlas/env.sh`. Vanda's CPU nodes have 72 cores (Lecture 9), so each node in a 256-rank job leaves 8 cores idle, and 256 ranks need 4 nodes.

**Why did we do it?** The rank counts 1, 2, 4, ..., 256 are powers of two, and the lecturer asked for scaling to at least 256 cores. Using 72 would give 72, 144, 216, 288 and no 256. Powers of two also give tidy process grids (16 by 16 in 2D, 8 by 8 by 4 in 3D) and clean log-scale plots.

**How does it work?** The idle cores do not hurt the measurement. They slightly help it, because the ranks on a node compete less for memory bandwidth. The plausible alternative is 72 per node with rank counts 72, 144, 216, 288, which would use all the cores and reach 288.

**What came out of it?** The choice determines the x-axis of every cluster plot and cannot be assessed until the jobs run. The per-node memory-bandwidth effect will be visible in the cluster hybrid figure by comparing the 64-ranks-per-node run with the 1-rank by 64-threads run. It cannot be read from the laptop data.

**How would I explain it to a classmate in two sentences?** We use 64 of each node's 72 cores so the rank counts are powers of two and reach exactly 256. The spare cores make each node a little less crowded, if anything.

### Assumption (i): the process grid shape is chosen by MPI unless we force it

**What is it?** When `--dims` is not given, `MPI_Dims_create` picks the most cube-like factorisation of the rank count: 4 by 4 for 16 ranks in 2D, 4 by 2 by 2 for 16 ranks in 3D, 2 by 2 by 2 for 8.

**Why did we do it?** Cube-like blocks have the smallest surface for their volume, so they send the fewest boundary values. The decomposition study then forces other shapes (`--dims 16,1` and so on) to measure how much that matters.

**How does it work?** For the 2048 by 2048 grid on 16 ranks, a 4 by 4 grid gives 512 by 512 blocks that send 4 x 512 = 2048 values per sweep. A 16 by 1 grid gives 128 by 2048 slabs that send 2 x 2048 = 4096 values. Twice the volume, but in two messages instead of four, and the slab's faces are contiguous in memory for one orientation and strided for the other.

**What came out of it?** The measured answer is in `decomp.csv` (Part E10). In 2D all five shapes finished within 15 percent of each other. In 3D the shape mattered through memory layout rather than volume: x-slabs took 0.420 s against 0.295 s for z-slabs with the same 73728 halo values per rank, and the automatic `MPI_Dims_create` choice (4x2x2, 0.298 s) was within 1 percent of the best.

**How would I explain it to a classmate in two sentences?** By default MPI cuts the grid into blocks as square as possible, which minimises the boundary to exchange. We also tried strips to see whether fewer, longer messages are faster.

### Assumption (j): the residual check is switched off in the Jacobi benchmarks

**What is it?** `--check-every 0` in every Jacobi timing row, so the only communication is the halo exchange. In the check-interval study it is set to 1, 10 and 100.

**Why did we do it?** Each check costs an extra halo exchange, a full pass over the block and one `MPI_Allreduce`, which is the one operation in which every process must meet. Separating it out lets us attribute the cost of the exchange and the cost of the global synchronisation to different columns.

**How does it work?** With a check every sweep (1), the all-reduce count equals the sweep count and the extra pass doubles the arithmetic. With every 100 sweeps the cost is 1 percent of that. The plausible range in practice is 10 to 100.

**What came out of it?** Part E11 measured it at 16 ranks: checking every sweep raised the Jacobi 2D time from 0.216 s to 0.566 s (0.345 s of it in all-reduce), checking every 10 sweeps gave 0.205 s and every 100 sweeps 0.203 s, both within noise of never checking. In real use one would check every 10 to 50 sweeps.

**How would I explain it to a classmate in two sentences?** The benchmark never stops to measure its own progress, so the only talking between processes is the boundary swap. We measured separately what each progress check costs.

### Assumption (k): double precision arithmetic

**What is it?** Every array holds 8-byte `double` values.

**Why did we do it?** It is the default for scientific computing and what the lecture examples use, and it lets the convergence study reach errors of 1e-7 without rounding noise.

**How does it work?** Single precision (4-byte `float`) halves the memory traffic, which for a memory-bound stencil roughly doubles the speed per sweep and halves the bytes exchanged. It also caps the attainable accuracy near 1e-7 relative, so the tighter tolerances in `validation.csv` would be unreachable. The plausible range is float, double, or mixed.

**What came out of it?** All timings would shift by up to a factor of two with float. Speedup and efficiency would change little, since compute and communication shrink together. The error columns below about 1e-6 would be rounding noise.

**How would I explain it to a classmate in two sentences?** We use 8-byte numbers, the scientific standard. Four-byte numbers would be up to twice as fast but could not reach the accuracies we test.

### Assumption (l): only the iteration loop is timed

**What is it?** `t_total` runs from a barrier just before the first sweep to a barrier just after the last. Filling f (`t_setup`, recorded separately) and the final residual and error computation (not recorded) are outside it. In the words of the writing guide's translation of the brief, the "system boundary" of the measurement is the iteration loop.

**Why did we do it?** The setup is a once-only cost that would otherwise distort short runs, and the diagnostics exist only to check correctness. A production solver would compute neither.

**How does it work?** `t_setup` in `strong.csv` is 0.026 s for the serial Jacobi 2D run against a loop time of 0.527 s, or 5 percent. On 16 ranks it is 0.007 s against 0.302 s. Including it would lower every efficiency by a few percent and penalise short runs most. The plausible range is loop only, loop plus setup, or whole program (including process start-up, which for 256 ranks can be seconds).

**What came out of it?** Reported speedups are for the loop. A reader who wants whole-run figures can add `t_setup` from the CSV. Process start-up time is not recorded anywhere.

**How would I explain it to a classmate in two sentences?** We time the repeated part of the program and record setup separately. Including setup would shave a few percent off the efficiencies, and we have not measured program start-up at all.

### Assumption (m): one GPU per MPI rank

**What is it?** The CUDA driver `src/cuda/mpi_cuda_main.cu` gives each process one GPU, chosen as the process's number within its node modulo the number of GPUs. The job scripts request `ngpus=1` with one rank and `ngpus=2` with two ranks.

**Why did we do it?** It is the model Lecture 10 recommends for MPI+CUDA: no device switching inside a process, and the same decomposition code as the CPU version.

**How does it work?** The alternative is one process driving several GPUs, which needs explicit device switching and streams. With one GPU per rank the halo exchange between two GPUs on the same node goes through the host memory unless the MPI library is CUDA-aware, which the `--cuda-aware` flag requests.

**What came out of it?** Nothing yet. The CUDA build has been compiled and run only through the CPU emulation layer in `src/cuda/cuda_compat.hpp`, where it reproduces the serial numbers (the `mpi_cuda_emul` rows in `validation.csv` have the same `l2_err` 0.8370888068897 as the serial row). The gpu_vs_cpu figure depends entirely on `results/cluster/gpu.csv`, which does not exist until `scripts/vanda/gpu.pbs` has been run on Vanda.

**How would I explain it to a classmate in two sentences?** Each process owns one GPU, the simplest arrangement. The GPU code is correct in emulation but has never run on real GPU hardware yet.

## C7. Sanity checks you can do on the data in two minutes

**What is it?** A short routine for convincing yourself that a CSV file is trustworthy before you believe any figure drawn from it.

**Why did we do it?** The figures are several steps removed from the measurements. If a row is bad (a run that did not converge, a process that was descheduled, a mismatch between implementations), the place to see it is the raw file.

**How does it work?** Open the file in any spreadsheet program or print it in a terminal:

```bash
cd poisson-mpi/results/local
column -s, -t < strong.csv | less -S        # aligned columns, scroll sideways with the arrow keys
```

First check correctness. In `validation.csv`, every row tagged `agree` with the same `solver`, `dim` and `nx` must show the same `iters` and the same `l2_err` and `linf_err` to at least the twelfth digit, whatever the `impl`, `ranks` or `exchange`. For the Jacobi 2D case (`nx` 300, 300 sweeps) the eight rows all read 0.83708880688971 for `l2_err`, differing only beyond that, and 1.7034276923218 for `linf_err`. For the CG rows the `rel_resid` may differ in its last digits (5.665e-10 serial against 5.660e-10 on 6 ranks) because global sums are added in a different order, while `iters` must be identical (432) and `l2_err` must agree to about ten digits. Any larger disagreement is a bug.

Second, recompute a speedup by hand. Take the serial row for a case and the parallel row you care about, both in `strong.csv`, and divide `t_total`. Serial Jacobi 2D: 0.527196. Eight ranks: 0.105227. Speedup 5.01, efficiency 5.01 / 8 = 0.63. If the analysis script's table says anything else, the script is wrong, not the data.

Third, learn the three signatures of a bad or misread row. A row with `tol` greater than 0 and `converged` 0 means the solver hit the `iters` cap without reaching the tolerance. None of our rows do this, and one that did should be rerun with a larger cap. A row whose `t_halo_max` is several times its `t_halo_mean` (0.2392 against 0.2267 is fine. 0.0745 against 0.0638 on the 8-rank weak-scaling repeat is a mild case. The first repeat of that run had 0.0152 against 0.0098) means one process was held up, and the minimum-of-repeats rule exists for exactly this. A parallel row whose `iters` differs from the serial row for the same CG problem would mean the parallel version is doing different arithmetic. Every CG row in `strong.csv` has 1458 or 222 iterations matching its serial row, as it should.

**What came out of it?** All 170 or so rows in the four existing files pass these checks. The only striking feature, the jump in `t_halo_max` and `t_allreduce_max` between 8 and 16 ranks, is explained by the machine (Section C5), not by the code.

**How would I explain it to a classmate in two sentences?** Open the CSV, check that every implementation gives the same error for the same problem, and divide two times to get a speedup yourself. If a run did not converge or one process waited far longer than the average, you will see it in the raw columns before it hides inside a figure.
# Part D. The folder and the code, file by file

This Part opens every file in the project and explains what it is for, how it works, and what it guarantees. Part B explained the methods. Here the question is "where in the code does that happen, and what do the lines actually say". Code is quoted from the files as they are. Where the code is unfinished or only partly tested, that is stated at the point where it matters.

A few words that recur throughout this Part, defined once here. A **source file** is a plain text file containing instructions for the computer in a programming language. Think of it as the recipe, as opposed to the cooked dish, which is the **binary** (the runnable program the compiler produces from the recipe). A **compiler** is the program that translates source files into a binary, the way a translator turns a recipe written in French into one your oven-ready English-speaking cook can follow. A **header file** (ending in `.hpp` or `.cuh`) is a source file that is meant to be pasted into other source files by the compiler. It holds shared definitions, like a shared appendix that several chapters refer to instead of each repeating it. A **shell script** (ending in `.sh` or `.pbs`) is a list of terminal commands saved in a file so they can be run in one go, like a checklist that the computer ticks off by itself.

## D1. Map of the folder

**What is it?** The project lives in one folder called `poisson-mpi`. Below is every file and sub-folder with its purpose in one line. The tree was taken from the folder itself at the time of writing.

```text
poisson-mpi/
├── Makefile                     recipe for turning the source files into the binaries
├── README.md                    short instructions: build, data, run, reproduce the figures
├── src/                         all C++ and CUDA source code
│   ├── common.hpp               problem definition, command-line options, result record, CSV writer
│   ├── kernels.hpp              the arithmetic: Jacobi sweep, operator, dot product, vector updates (plain and masked)
│   ├── gridio.hpp               the satellite grid file, per-rank block loading, the pixel mask, the fill errors
│   ├── solver.hpp               the Jacobi and conjugate-gradient loops, written once for every backend and both problems
│   ├── decomp.hpp               MPI domain decomposition, the four halo-exchange strategies, MPI-IO output of the filled field
│   ├── serial_main.cpp          driver for the serial binary and the OpenMP-only binary
│   ├── mpi_main.cpp             driver for the MPI binary and the MPI+OpenMP binary
│   └── cuda/
│       ├── cuda_compat.hpp      lets the CUDA code compile and run on a machine with no GPU (emulation)
│       ├── gpu_kernels.cuh      the CUDA kernels (GPU versions of kernels.hpp, plain and masked)
│       └── mpi_cuda_main.cu     driver for the MPI+CUDA binary
├── tools/
│   ├── gridtool.cpp             C++ data tool: downloaded text grids -> grid file; info, slices, differences
│   ├── md2pdf.sh, *.css         Markdown-to-PDF build for the report and this handbook
│   └── handbook_source.md       the Markdown source of this handbook
├── scripts/
│   ├── data/download.sh         downloads the daily satellite maps, the land mask and the reference from NOAA
│   ├── local/run_all.sh         runs every experiment on a laptop or workstation, writes results/local/*.csv
│   ├── local/maps.sh            draws the observed / filled maps of one day
│   ├── atlas/env.sh             cluster settings the student must edit (queue name, ranks per node, modules)
│   ├── atlas/gen_jobs.sh        writes one PBS job file per experiment into jobs/, optionally submits them
│   └── vanda/
│       ├── gpu.pbs              one-GPU job: GPU against a full CPU node, both problems
│       └── gpu_multi.pbs        two-GPU job
├── analysis/
│   ├── summarize.cpp            C++ tool: reads the CSVs, writes tables.md, summary.txt and .dat plot data
│   ├── plots.gp                 gnuplot script: draws every PNG figure from the .dat files
│   └── maps.gp                  gnuplot script: the satellite maps
├── data/                        the satellite grids (downloaded and built locally; not in the submission)
│   ├── raw/india/               365 daily text files, the mask and 365 reference files from NOAA
│   ├── india_2025.grid          the stacked year as the solver reads it (801 x 534 x 365, 625 MB)
│   └── india_2025_ref.grid      the reference analysis on the same grid
├── bin/                         compiled binaries (created by make; not part of the source)
│   ├── poisson_serial           serial baseline
│   ├── poisson_omp              OpenMP threads only
│   ├── poisson_mpi              MPI only
│   ├── poisson_mpi_omp          MPI + OpenMP
│   ├── poisson_mpi_cuda_emul    the CUDA source compiled as ordinary C++ (testing only)
│   ├── poisson_mpi_cuda         MPI + CUDA (only exists after "make cuda" on a machine with nvcc)
│   ├── gridtool                 the data tool built from tools/gridtool.cpp
│   └── summarize                the analysis tool built from analysis/summarize.cpp by "make figures"
├── jobs/                        created by scripts/atlas/gen_jobs.sh when it is run (29 PBS files; not kept in the repository)
├── results/
│   ├── local/                   CSVs and run.log written by scripts/local/run_all.sh on the laptop
│   ├── fields/                  the filled temperature fields (grid files) written by the data runs
│   ├── figures/                 PNG figures, tables.md, summary.txt and *.dat plot data from make figures
│   └── cluster/                 empty until the student runs the cluster jobs; the jobs write here
├── report/                      the project report (report.md and report.pdf, built by build.sh)
├── tools/                       Markdown-to-PDF build script, stylesheets, and the handbook's Markdown source
└── handbook/                    this handbook (handbook.pdf)
```

**Why did we do it?** Separating source, scripts, analysis and results into their own folders means a reader can tell at a glance which files are written by a person and which are generated. Everything under `bin/`, `jobs/` and `results/` is generated and can be deleted and recreated. Everything else is hand-written.

**How does it work?** The Makefile reads `src/` and writes `bin/`. The scripts run the binaries and write `results/*/*.csv`. `make figures` builds `bin/summarize`, which reads those CSVs and writes tables and plot data into `results/figures/`, and then gnuplot draws the PNG files there. The report and the handbook pull their figures from `results/figures/`. The chain only flows in that direction.

**What came out of it?** `results/local/` holds the ten CSV files described in D9 and `run.log`. `results/cluster/` is empty because no cluster job has been submitted yet. `bin/poisson_mpi_cuda` does not exist because this laptop has no `nvcc`.

**How would I explain it to a classmate in two sentences?** Source code in `src`, scripts that run it in `scripts`, the thing that draws the graphs in `analysis`, and everything the scripts produce in `results`. If a folder is named after an output you can regenerate it. If it is named after an input you must not lose it.

## D2. How to build: the Makefile

**What is it?** The Makefile is the recipe that tells the tool `make` how to turn source files into binaries, and which binaries depend on which sources so that only what changed is rebuilt. It is like a kitchen checklist that says "the sauce needs the stock, the stock needs the bones, and if only the bones changed, remake the stock and the sauce, not the bread".

**Why did we do it?** There are six binaries built from five source files with four different compilers or compiler settings. Typing those commands by hand is error-prone, and the lecturer's Lecture 9 slides explicitly recommend a makefile or CMake for exactly this reason.

**How does it work?** The file starts with variables, which are names for pieces of text that can be overridden from the command line:

```makefile
CXX      ?= g++
MPICXX   ?= mpicxx
NVCC     ?= nvcc
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -march=native
CUDA_ARCH ?= sm_70
```

`?=` means "use this value unless the user already gave one", so `make all CXX=clang++` replaces the compiler. `CXX` is the ordinary C++ compiler, `MPICXX` is the MPI wrapper (a small program that calls the real compiler with the extra paths MPI needs added), and `NVCC` is NVIDIA's CUDA compiler. The flags in `CXXFLAGS` mean: `-O3` asks for the highest level of optimisation (the compiler rearranges the arithmetic to run fast and does not change the results beyond the last digit of floating-point sums). `-std=c++17` selects the 2017 edition of the C++ language, which this code uses. `-Wall -Wextra` turns on warnings about suspicious code. `-march=native` tells the compiler it may use every instruction the current CPU has, including vector instructions that process several numbers at once.

```makefile
UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
  OMPFLAGS ?= -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
  OMPLIBS  ?= -L/opt/homebrew/opt/libomp/lib -lomp
  # Apple clang does not accept -march=native on arm64
  CXXFLAGS := $(filter-out -march=native,$(CXXFLAGS)) -mcpu=native
else
  OMPFLAGS ?= -fopenmp
  OMPLIBS  ?= -fopenmp
endif
```

`uname` prints the operating system name, which on a Mac is `Darwin`. Apple's compiler spells the "use everything this CPU has" flag `-mcpu=native` rather than `-march=native`, so the Mac branch swaps one for the other. OpenMP (the threading system explained in Part B) is switched on with `-fopenmp` on Linux and that single flag also links the runtime library. Apple's compiler understands the `#pragma omp` lines but does not ship the runtime, so on a Mac the flag is passed through to the preprocessor with `-Xpreprocessor -fopenmp`, the include folder of the Homebrew `libomp` package is added with `-I`, and the library is linked explicitly with `-L... -lomp`. That is the "libomp dance" the lecturer mentioned in Lecture 9.

```makefile
MPI_COMPILE := $(shell $(MPICXX) --showme:compile 2>/dev/null || $(MPICXX) -compile_info 2>/dev/null | cut -d' ' -f2-)
MPI_LINK    := $(shell $(MPICXX) --showme:link 2>/dev/null    || $(MPICXX) -link_info 2>/dev/null    | cut -d' ' -f2-)
```

`mpicxx --showme:compile` asks the Open MPI wrapper to print the include flags it would silently add (on this laptop: `-I/opt/homebrew/Cellar/open-mpi/5.0.11/include`), and `--showme:link` prints the library flags. The `||` means "if that fails, try the MPICH spelling `-compile_info` instead". These are needed only for the CUDA build, because `nvcc` is not an MPI wrapper and has to be told where MPI lives.

```makefile
HDRS := src/common.hpp src/kernels.hpp src/solver.hpp
BIN  := bin

.PHONY: all serial omp mpi mpi_omp cuda cuda_emul clean

all: serial omp mpi mpi_omp

$(BIN):
	mkdir -p $(BIN)
```

`.PHONY` declares that names like `all` and `clean` are commands, not files to be produced. `all` builds the four CPU binaries. The `$(BIN)` rule creates the `bin` folder.

```makefile
$(BIN)/poisson_serial: src/serial_main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

$(BIN)/poisson_omp: src/serial_main.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -Isrc $< -o $@ $(OMPLIBS)
```

Each rule says that the file on the left depends on the files on the right, and if any of them is newer the indented command runs. `$<` stands for the first dependency (the `.cpp` file) and `$@` for the target. The vertical bar before `$(BIN)` marks an "order-only" dependency: the folder must exist first, but its timestamp changing must not trigger a rebuild. The two rules above compile the same file, `serial_main.cpp`. The only difference is `$(OMPFLAGS)`. Without it the compiler ignores every `#pragma omp` line and the symbol `_OPENMP` is undefined, so the `#ifdef _OPENMP` branches in the source are skipped and the result is a purely serial program. With it, the pragmas become real threads. One source file, two binaries.

```makefile
$(BIN)/poisson_mpi: src/mpi_main.cpp src/decomp.hpp $(HDRS) | $(BIN)
	$(MPICXX) $(CXXFLAGS) -Isrc $< -o $@

$(BIN)/poisson_mpi_omp: src/mpi_main.cpp src/decomp.hpp $(HDRS) | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -Isrc $< -o $@ $(OMPLIBS)
```

The same trick applies to `mpi_main.cpp`: compiled with the MPI wrapper alone it is pure MPI, compiled with the OpenMP flags added it is MPI+OpenMP.

```makefile
$(BIN)/poisson_mpi_cuda: src/cuda/mpi_cuda_main.cu src/cuda/gpu_kernels.cuh src/cuda/cuda_compat.hpp src/decomp.hpp $(HDRS) | $(BIN)
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -Isrc $(MPI_COMPILE) $< -o $@ $(MPI_LINK)

$(BIN)/poisson_mpi_cuda_emul: src/cuda/mpi_cuda_main.cu src/cuda/gpu_kernels.cuh src/cuda/cuda_compat.hpp src/decomp.hpp $(HDRS) | $(BIN)
	$(MPICXX) $(CXXFLAGS) -x c++ -Isrc $< -o $@
```

The real GPU build uses `nvcc` with `-arch=sm_70` by default, where `sm_70` names the hardware generation (V100). The student must pass `CUDA_ARCH=sm_80` for an A100 or `sm_90` for an H100. The emulation build compiles the very same `.cu` file with the ordinary MPI wrapper, forcing it to be treated as C++ with `-x c++`. How that can possibly work is explained in D8. Finally `clean` deletes `bin/`.

**What came out of it?** On this laptop `make all CXX=clang++` produced the four CPU binaries with no warnings, and `make cuda_emul CXX=clang++` produced the emulation binary. `make cuda` has not been run because there is no `nvcc` here.

**How would I explain it to a classmate in two sentences?** The Makefile builds six programs from five source files by varying the compiler and the flags, and the OpenMP-on and OpenMP-off binaries come from the same source because the compiler simply ignores the threading pragmas when the flag is absent. Run `make all` on a CPU machine and `make cuda CUDA_ARCH=...` on a GPU machine.

## D3. src/common.hpp: the problem, the options and the result record

**What is it?** `common.hpp` is the shared appendix every other file includes. It defines the problem being solved, the list of command-line options, how the grid is laid out in memory, how the exact solution is computed for checking, how the grid is split between processes, and how one line of results is written to a CSV file. A **CSV file** is a plain text table where each line is a row and commas separate the columns, the simplest format a spreadsheet can open.

**Why did we do it?** Every variant (serial, OpenMP, MPI, MPI+OpenMP, MPI+CUDA) must solve exactly the same problem with exactly the same options and write results in exactly the same format, or the comparisons would be meaningless. Putting these in one header makes it impossible for the variants to drift apart.

**How does it work?** The file opens with a comment that is the most compact statement of the whole project:

```cpp
// Problem solved:   -laplace(u) = f   on the unit square (dim=2) or unit cube (dim=3)
//                   u = 0 on the boundary (Dirichlet condition)
// Manufactured solution (used for validation):
//                   u(x,y,z) = exp(x) sin(pi x) sin(pi y) [sin(pi z)]
//                   f = -laplace(u) = exp(x) sin(pi y) [sin(pi z)] *
//                                     ( (dim*pi^2 - 1) sin(pi x) - 2 pi cos(pi x) )
```

Everything sits inside `namespace poisson { ... }`. A **namespace** is a label attached to a group of names so they cannot collide with names from other libraries, like putting a surname after a first name.

The options live in a **struct**, which is a bundle of named values travelling together, like a form with labelled boxes:

```cpp
struct Params {
    int dim = 2;                          // 2 or 3
    long n[3] = {256, 256, 1};            // global interior points per dimension (nz=1 for 2D)
    long local_n = 0;                     // weak scaling: if >0, n_d = local_n * dims_d
    std::string solver = "jacobi";        // jacobi | cg
    std::string exchange = "isend";       // sendrecv | isend | overlap | rma
    int dims[3] = {0, 0, 0};              // process grid (0 = let MPI_Dims_create decide)
    long iters = 1000;                    // maximum iterations
    double tol = 0.0;                     // relative residual tolerance (0 = run exactly `iters` sweeps)
    int check_every = 0;                  // jacobi: compute residual every k iterations (0 = never)
    int threads = 0;                      // OpenMP threads override (0 = OMP_NUM_THREADS)
    std::string csv;                      // append a results row to this file
    std::string tag;                      // free text copied to the CSV row
    bool verbose = false;
    bool cuda_aware = false;              // mpi_cuda: pass device pointers straight to MPI
};
```

Each field is one command-line option, and the value after `=` is its default. `dim` chooses square or cube. `n` holds the number of interior grid points in x, y and z (for a 2D run `n[2]` is forced to 1). `local_n` is the weak-scaling switch: if set, the global size is computed from the number of processes so that every process always owns `local_n` points per dimension. `solver` picks Jacobi or conjugate gradient. `exchange` picks the halo strategy. `dims` forces the shape of the process grid, for example 4 by 4 instead of 16 by 1. `iters` and `tol` together define when to stop: with `tol = 0` the code runs exactly `iters` sweeps (a benchmark), with `tol > 0` it stops when the relative residual drops below `tol` (a real solve). `check_every` controls how often Jacobi pays for the global residual check. `csv` and `tag` say where to record the result and how to label it.

The option parser is `parse_args`. Its helper `parse_list` turns `"256"` into three equal sizes and `"256,128"` into `{256, 128, fill}`:

```cpp
inline void parse_list(const std::string& s, long* out, int maxn, long fill) {
    // "256" -> {256,256,256}; "256,128" -> {256,128,fill}
    std::vector<long> v;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t c = s.find(',', pos);
        if (c == std::string::npos) c = s.size();
        v.push_back(std::atol(s.substr(pos, c - pos).c_str()));
        pos = c + 1;
    }
    if (v.size() == 1) { for (int d = 0; d < maxn; ++d) out[d] = v[0]; return; }
    for (int d = 0; d < maxn; ++d) out[d] = d < (int)v.size() ? v[d] : fill;
}
```

A `std::vector` is a resizable list of values. The loop finds each comma, converts the text between commas to a number with `std::atol`, and collects them. `parse_args` then walks the arguments two at a time (`--name value`), fills the struct, and rejects anything it does not recognise with a usage message. It also enforces that `dim` is 2 or 3, that `solver` is `jacobi` or `cg`, and that `exchange` is one of the four known strategies.

The memory layout is captured by `Local`:

```cpp
struct Local {
    int nx = 0, ny = 0, nz = 0;       // interior points owned by this rank
    long ox = 0, oy = 0, oz = 0;      // global index offset of local interior point (1,1,1)
    long sx = 1, sy = 0, sz = 0;      // strides
    size_t ntot = 0;                  // number of stored doubles including halos
    double hx = 0, hy = 0, hz = 0;    // grid spacings (hz unused in 2D)
    double ihx2 = 0, ihy2 = 0, ihz2 = 0; // 1/h^2 per direction (ihz2 = 0 in 2D)
    double diag = 0;                  // diagonal of the operator: 2*(ihx2+ihy2+ihz2)
    inline size_t idx(int i, int j, int k) const { return (size_t)k * sz + (size_t)j * sy + (size_t)i; }
};
```

A grid is a 2D or 3D object, but computer memory is a single long row of boxes. A **stride** is how many boxes you must skip to move one step in a given direction, like saying "in this car park, the next row is 20 spaces further along". Here `sx = 1` (moving one step in x is the next box), `sy = nx + 2` (one step in y skips a whole row, including the two halo cells at its ends) and `sz = (nx + 2)(ny + 2)` (one step in z skips a whole layer). The function `idx` turns a triple into one box number. `inline` is a hint that the function is tiny and should be pasted in place rather than called, and `const` promises the function does not modify the struct.

A worked example: suppose a rank owns `nx = 4`, `ny = 3`, `nz = 1` interior points (a 2D block, since `nz = 1`). Then `sy = nx + 2 = 6`, `sz = (nx + 2)(ny + 2) = 6 x 5 = 30`, and `ntot = 6 x 5 x 3 = 90` boxes, because every dimension gets one halo layer on each side, including z. The interior cell `(i, j, k) = (2, 1, 1)` sits at box `1 x 30 + 1 x 6 + 2 = 38`. Its left neighbour is box 37, its right neighbour box 39, the cell below it (j = 0, a halo cell) is box 32 and the cell above (j = 2) is box 44. The code never computes neighbours by calling `idx` again. It just adds or subtracts `1`, `sy` or `sz`, which is why the strides are stored.

`make_local` fills a `Local` from the global sizes:

```cpp
    L.sx = 1; L.sy = nx + 2; L.sz = (long)(nx + 2) * (ny + 2);
    L.ntot = (size_t)(nx + 2) * (ny + 2) * (nz + 2);
    L.hx = 1.0 / (gn[0] + 1); L.hy = 1.0 / (gn[1] + 1);
    L.ihx2 = 1.0 / (L.hx * L.hx); L.ihy2 = 1.0 / (L.hy * L.hy);
    if (p.dim == 3) { L.hz = 1.0 / (gn[2] + 1); L.ihz2 = 1.0 / (L.hz * L.hz); }
    else            { L.hz = 1.0; L.ihz2 = 0.0; }
    L.diag = 2.0 * (L.ihx2 + L.ihy2 + L.ihz2);
```

The spacing in each direction is one over the number of intervals, `1/(n+1)`. The coefficient `ihz2` is set to zero in 2D. This is the trick that lets one set of 3D kernels serve 2D as well: the z-neighbours are multiplied by zero and vanish. The diagonal of the operator is `2/hx^2 + 2/hy^2 + 2/hz^2`.

The exact solution and right-hand side are evaluated at a global index:

```cpp
inline double exact_at(const Params& p, const Local& L, long gi, long gj, long gk) {
    const double x = gi * L.hx;
    double v = std::exp(x) * std::sin(PI * x) * std::sin(PI * gj * L.hy);
    if (p.dim == 3) v *= std::sin(PI * gk * L.hz);
    return v;
}
inline double rhs_at(const Params& p, const Local& L, long gi, long gj, long gk) {
    const double x = gi * L.hx;
    double v = std::exp(x) * std::sin(PI * gj * L.hy);
    if (p.dim == 3) v *= std::sin(PI * gk * L.hz);
    return v * ((p.dim * PI * PI - 1.0) * std::sin(PI * x) - 2.0 * PI * std::cos(PI * x));
}
```

These are the formulas from the facts sheet. The comment in the file explains why there is an `exp(x)`: a plain product of sines happens to be an eigenvector of the discrete Laplacian, which means conjugate gradient would finish in one step and the benchmark would be meaningless. The `exp(x)` factor removes that accident. (This was discovered during testing: the first version of the code used plain sines and CG converged in exactly one iteration on every grid.)

Splitting `n` points among `P` processes is done by `partition`:

```cpp
inline void partition(long n, int P, int coord, int& count, long& offset) {
    long base = n / P, rem = n % P;
    count = (int)(base + (coord < rem ? 1 : 0));
    offset = coord * base + (coord < rem ? coord : rem);
}
```

With `n = 10` and `P = 4`: `base = 2`, `rem = 2`. Process 0 gets 3 points starting at offset 0, process 1 gets 3 starting at 3, process 2 gets 2 starting at 6, process 3 gets 2 starting at 8. The first `rem` processes take one extra point each so the sizes differ by at most one. The `&` on `count` and `offset` means the function writes its answers back into the caller's variables.

The result of a run is another struct, `Result`, with the implementation name, the rank count, the thread count, the process-grid string, iteration count, convergence flag, residual, two error norms, and the timers. `write_csv` appends it as one row:

```cpp
    bool header = true;
    if (FILE* f = std::fopen(p.csv.c_str(), "r")) { std::fseek(f, 0, SEEK_END); header = std::ftell(f) == 0; std::fclose(f); }
    FILE* f = std::fopen(p.csv.c_str(), "a");
```

It opens the file for reading, jumps to the end, and if the position is zero the file is empty and a header line is written first. Then it reopens in append mode (`"a"`) and prints the row. The columns, in order, are:

| column | meaning |
|---|---|
| impl | which binary: serial, omp, mpi, mpi_omp, mpi_cuda or mpi_cuda_emul |
| dim | 2 or 3 |
| nx, ny, nz | global interior grid points per dimension (nz = 1 in 2D) |
| solver | jacobi or cg |
| exchange | halo strategy requested |
| ranks | number of MPI processes |
| threads | OpenMP threads per process |
| dims | process grid, for example 4x4x1 |
| iters | iterations actually performed |
| converged | 1 if the tolerance was reached, 0 otherwise |
| rel_resid | final relative residual, norm(f - A u) / norm(f) |
| l2_err | discrete L2 norm of (u minus exact), printed with 17 digits |
| linf_err | largest absolute error at any point |
| t_total | wall time of the iteration loop, seconds (longest rank) |
| t_setup | time to fill f and compute norm(f) |
| t_compute_max, t_halo_max, t_allreduce_max | time in kernels, in halo exchange, in MPI_Allreduce: worst rank |
| t_compute_mean, t_halo_mean, t_allreduce_mean | the same, averaged over ranks |
| check_every | the residual-check interval that was in force |
| tol | the tolerance requested |
| hostname | the machine the run happened on |
| tag | the free-text label given on the command line |

`print_result` prints the same information on three lines for the terminal.

**What came out of it?** Every CSV row in `results/local/` has exactly these 27 columns. Because `l2_err` is printed with `%.17g` (seventeen significant digits), two runs whose arithmetic is identical will produce identical text, which is how the validation table in Part E can claim bit-level agreement.

**How would I explain it to a classmate in two sentences?** `common.hpp` is the contract: the problem, the options, the memory layout and the result format that every version of the program must share. If you want to change what the code solves or how it reports, this is the one file to touch.

## D4. src/kernels.hpp: the arithmetic

**What is it?** `kernels.hpp` holds the small loops that do all of the actual arithmetic: one Jacobi sweep, applying the operator A, computing the residual, the dot product, two vector updates, filling the right-hand side, and measuring the error. A **kernel** in this sense is a short, hot loop that is run millions of times, like the one step on an assembly line that is repeated for every item.

**Why did we do it?** The serial, OpenMP, MPI and MPI+OpenMP binaries all use these exact loops. Keeping them in one place means a speed-up or a bug fix applies everywhere, and it guarantees the serial baseline does the same arithmetic as the parallel versions, which is what makes the speedup numbers fair.

**How does it work?** Every kernel works on a sub-range of the block:

```cpp
struct Range { int i0, i1, j0, j1, k0, k1; };
inline Range full(const Local& L) { return {1, L.nx + 1, 1, L.ny + 1, 1, L.nz + 1}; }
```

A `Range` is a box of cells from `i0` up to but not including `i1`, and likewise in j and k. `full` is the whole interior: indices 1 to nx inclusive, because index 0 and index nx+1 are the halo. Working on sub-ranges is what allows the "overlap" exchange to compute the inner part of the block while messages are still travelling.

The Jacobi sweep:

```cpp
inline void jacobi_sweep(const Local& L, const double* __restrict u, double* __restrict un,
                         const double* __restrict f, Range r) {
    const long sy = L.sy, sz = L.sz;
    const double ihx2 = L.ihx2, ihy2 = L.ihy2, ihz2 = L.ihz2, inv_diag = 1.0 / L.diag;
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = r.k0; k < r.k1; ++k)
        for (int j = r.j0; j < r.j1; ++j) {
            const size_t row = L.idx(0, j, k);
#pragma omp simd
            for (int i = r.i0; i < r.i1; ++i) {
                const size_t c = row + i;
                un[c] = (f[c] + ihx2 * (u[c - 1] + u[c + 1])
                              + ihy2 * (u[c - sy] + u[c + sy])
                              + ihz2 * (u[c - sz] + u[c + sz])) * inv_diag;
            }
        }
}
```

Line by line. `const double* u` is a **pointer**: the address of the first box of the array in memory, like the street number where a row of houses begins, and the code walks along it. `__restrict` is a promise to the compiler that `u`, `un` and `f` do not overlap in memory, which lets it reorder loads and stores freely. Without the promise it would have to assume that writing `un[c]` might change `u[c+1]`. The four local copies of the strides and coefficients save the compiler from reloading them through the struct on every iteration. `inv_diag` turns the division by `diag` into a multiplication, which is cheaper.

The line `#pragma omp parallel for collapse(2) schedule(static)` is an OpenMP **pragma**: an instruction to the compiler that is not part of the language and is ignored if OpenMP is off, like a sticky note on a recipe that only the sous-chef reads. `parallel for` says "split the iterations of the following loop among threads". `collapse(2)` merges the k and j loops into one longer loop before splitting, so that in 2D (where the k loop has a single iteration) there is still plenty to share out. `schedule(static)` gives each thread a fixed, equal-sized contiguous chunk decided up front, which is the right choice when every iteration costs the same. The inner `#pragma omp simd` asks the compiler to process several consecutive `i` values with one vector instruction (SIMD: single instruction, multiple data, the concept from Lecture 7). The loop body is exactly the facts-sheet formula: the new value at the centre is the right-hand side plus the weighted sum of the six neighbours, divided by the diagonal. In 2D `ihz2` is zero, so the z line contributes nothing and `u[c - sz]` and `u[c + sz]` are the untouched zero halo layers.

`apply_A` has the same shape but computes `Ap = diag * p - weighted neighbours`, which is the operator (the discrete negative Laplacian) applied to a vector, needed by conjugate gradient. `residual` computes `res = f - A u` over the whole interior and returns the local sum of squares. Its pragma includes `reduction(+:sum)`, which tells OpenMP that every thread keeps its own `sum` and the pieces are added at the end, avoiding the race where two threads update one variable at the same moment. `dot` is a plain sum of products with the same reduction clause. `axpy` computes `y = y + alpha * x` (the name is the classical linear-algebra abbreviation for "a times x plus y"), and `xpay` computes `p = r + beta * p`, the direction update in CG. `init_problem` fills `f` by calling `rhs_at` at the global index `ox + i`, which is the local index shifted by this rank's offset. `error_norms` compares `u` with `exact_at` and returns the sum of squared errors and the largest absolute error, using two reductions, `+` for the sum and `max` for the maximum.

The last part of the file supports overlap:

```cpp
template <class Fn>
inline void for_each_boundary_slab(const Local& L, Fn fn) {
    const int nx = L.nx, ny = L.ny, nz = L.nz;
    // x-faces: full y,z extent
    fn(Range{1, 2, 1, ny + 1, 1, nz + 1});
    if (nx > 1) fn(Range{nx, nx + 1, 1, ny + 1, 1, nz + 1});
    if (nx > 2) {
        // y-faces: x in [2,nx)
        fn(Range{2, nx, 1, 2, 1, nz + 1});
        if (ny > 1) fn(Range{2, nx, ny, ny + 1, 1, nz + 1});
        if (ny > 2 && nz > 1) {
            // z-faces: x in [2,nx), y in [2,ny)
            fn(Range{2, nx, 2, ny, 1, 2});
            if (nz > 1) fn(Range{2, nx, 2, ny, nz, nz + 1});
        }
    }
}
inline Range deep_interior(const Local& L) {
    if (L.nz == 1) return Range{2, L.nx, 2, L.ny, 1, 2};       // 2D: a single z-layer
    return Range{2, L.nx, 2, L.ny, 2, L.nz};
}
```

A **template** is a function written with a placeholder type (`Fn`) that the compiler fills in for whatever is passed, so the same function can accept any callable. Think of a form letter with a blank for the recipient. Here `fn` is whatever kernel the solver wants applied, wrapped in a lambda (see D5). The cells that touch a halo form the outer layer of the interior. The function visits that layer as a set of slabs chosen so that no cell is counted twice. The picture for a 2D block with `nx = 6`, `ny = 4` (halo cells marked `h`):

```text
      i:   0   1   2   3   4   5   6   7
 j=5       h   h   h   h   h   h   h   h      halo row (owned by the neighbour above)
 j=4       h   X   Y   Y   Y   Y   X   h      X = x-face slabs: i=1 and i=nx=6, all j from 1 to 4
 j=3       h   X   .   .   .   .   X   h      Y = y-face slabs: j=1 and j=ny=4, but only i from 2 to 5
 j=2       h   X   .   .   .   .   X   h      . = deep interior: Range{2,6, 2,4, 1,2}, i=2..5, j=2..3
 j=1       h   X   Y   Y   Y   Y   X   h
 j=0       h   h   h   h   h   h   h   h      halo row (owned by the neighbour below)
```

The x-face slabs take the full height, so the corners belong to them. The y-face slabs therefore start at `i = 2` and stop before `i = nx`, which is exactly the `Range{2, nx, ...}` in the code. In 3D the z-face slabs are further restricted to `j` from 2 to `ny - 1` so they do not re-visit the y-faces either. Together X, Y, (Z) and the deep interior cover every interior cell exactly once. The `if` guards handle thin blocks: when `nx` is 1 or 2 the two x-faces already cover everything and the other slabs are skipped. `deep_interior` is the dotted region, and in 2D it keeps `k` fixed at 1. `has_deep_interior` returns false when the box is empty, in which case the solver falls back to a plain exchange followed by a full sweep (see D5).

**What came out of it?** These loops are the whole cost of the program. The analysis in Part E shows the serial time per iteration growing linearly with the number of points, which is what you expect from loops that visit each cell once and do a fixed handful of operations there.

**How would I explain it to a classmate in two sentences?** Each kernel is one pass over a box of grid cells, with OpenMP sticky notes telling the compiler how to share the rows among threads. The boundary-slab helper carves the outer layer of the box into non-overlapping pieces so the inside can be computed while the edges wait for the neighbours' data.

A second family of kernels in the same file serves the satellite data: `jacobi_sweep_m`, `apply_A_m` and `residual_m`, where the `_m` stands for masked. They take one extra argument, a pointer to an array of bytes with one entry per cell, the mask: 0 for a missing pixel to be filled, 1 for an observed pixel, 2 for land (or outside the box), 3 for an observed pixel deliberately hidden for testing. The masked Jacobi sweep visits every cell of the range, skips the cell unless its mask is 0 or 3, and then looks at the mask of each of its four (2D) or six (3D) neighbours: a land neighbour contributes nothing and is left out of the diagonal, every other neighbour contributes its weight times its value. The diagonal is therefore not a constant but the sum of the weights of the neighbours that exist, recomputed on the fly, which is what makes the coast an insulated wall: a pixel with land on one side averages over three neighbours instead of four. If a pixel has no active neighbour at all (a one-pixel pond) its row is replaced by the identity, so it keeps the value zero, which after adding back the mean temperature means "the mean". Observed and land cells always hold zero in the solver arrays, because their contribution has been moved to the right-hand side by `finish_data` in `src/gridio.hpp`, so the dot products, vector updates and halo exchange do not need masked versions at all. The price of the mask is six extra byte loads and a branch per cell, which is why the data runs in Part E are a little slower per cell than the manufactured ones.

## D4a. src/gridio.hpp: the satellite grid, the mask and the fill errors

**What is it?** A header of about 250 lines that owns everything specific to the satellite data: the layout of the grid file, the reading of one rank's block from it, the classification of every cell into the four mask codes, the right-hand side of the gap-filling system, the error measures on the hidden pixels, and the writing of the filled field. The solver and the drivers call it; it does not call them.

**Why did we do it?** The gap filling must work identically in the serial, MPI, MPI+OpenMP and MPI+CUDA programs, and the only way to guarantee that is to have one copy of the data logic. Keeping it out of `solver.hpp` also keeps the solver readable: the solver sees a block of numbers and a mask, nothing about satellites.

**How does it work?** The file format is a struct `GridHeader` of exactly 84 bytes (checked by a `static_assert`): an 8-byte magic string `SSTGRD01`, the three sizes `nx, ny, nt`, the longitude and latitude of the first pixel and the spacing in degrees, the first date as text, the variable name, and four spare integers. After the header come `nt * ny * nx` 32-bit floats in day-major order. Latitude ascends, which `gridtool convert` guarantees by reversing the rows if the source had north first. The function `grid_offset(h, t, j, i)` turns a (day, row, column) triple into a byte offset, and `read_block` uses it to read one rank's block, including its halo layer, with one `fseek` and one `fread` per row: a rank that owns columns `ox .. ox+nx-1` reads columns `ox-1 .. ox+nx` (clamped to the box) for each of its rows and days, and anything outside the box becomes the land value. Reading the halo from the file means the mask of the halo cells is known locally and never has to be sent, and it means no rank ever needs more memory than its own block plus a one-cell rim.

`load_data` calls `read_block` for the data and, if `--ref` was given, for the reference, then classifies every cell: land value gives 2, NaN gives 0, a number gives 1, and a number whose pixel is selected by the hold-out rule gives 3. The rule is `hash01(gi / B, gj / B, gk, seed) < holdout`, where `hash01` is a small deterministic hash (a variant of splitmix64) of the global block indices, so that all pixels of the same B by B patch on the same day get the same decision, and so that the decision depends only on the pixel's position on the globe and the seed, never on how many ranks are running. That last point is what makes the hidden-pixel error a valid agreement test across rank counts. `finish_data` then computes the mean observed temperature with one global sum (the function takes a "sum" callback, which is `MPI_Allreduce` in the parallel programs and the identity in the serial one) and builds the right-hand side: for each cell to be solved, the sum over observed neighbours of weight times (observation minus mean). `data_local_errors` walks the interior after the solve and accumulates, for the hidden pixels, the squared error, the absolute error, the maximum error and the squared error of the baseline "mean" fill, and, for the reference, the squared differences on filled and on observed pixels; the solver reduces those sums across ranks and turns them into the RMSE, MAE and maximum columns of C3. `compose_output` assembles the filled field (observation where observed, fill plus mean where filled, land value on land) and `write_grid_serial` writes it; the MPI programs write theirs through `write_grid_mpi` in `decomp.hpp`, which uses a subarray datatype as a file view so that every rank writes its block into the right place of one file in a single collective call.

**What came out of it?** The two filled fields in `results/fields/` and every number in the gap-filling tables. The agreement test of E19, in which serial, MPI on several rank counts, OpenMP and the emulated CUDA program report the same hidden-pixel error to four decimals, is the evidence that the block reading, the hash and the masked kernels are right.

**How would I explain it to a classmate in two sentences?** This header knows how the satellite file is laid out, reads each process's own tile straight out of it, and labels every pixel as missing, observed, land or hidden-for-testing. It also moves the observed neighbours into the right-hand side, so the rest of the solver can treat the data problem as an ordinary Poisson problem with a few cells switched off.

## D5. src/solver.hpp: the Jacobi and CG loops

**What is it?** `solver.hpp` contains the two iterative methods, written once. It does not know whether it is running on one process or two hundred and fifty-six. Everything that depends on that is delegated to a small helper object called `Comm`.

**Why did we do it?** The loop logic for Jacobi and conjugate gradient is subtle enough (two buffers to swap, two global sums per CG step, careful timer placement) that having one copy for serial and another for MPI would be an invitation to make them differ. With one copy, the serial and parallel binaries are provably doing the same steps.

**How does it work?** The idea is a plug socket. The solver needs a few services: the time of day, "refresh the halos of this array", "sum this number over all processes", "find the maximum over all processes", and a barrier. The file's header comment lists them:

```cpp
//   C.wtime()                 wall clock
//   C.exchange(u)             refresh all halos of u (blocking from the caller's view)
//   C.can_overlap()           true if start/wait may be used to hide communication
//   C.start(u) / C.wait()     split-phase exchange for the overlap mode
//   C.sum(x) / C.max(x)       global reductions
//   C.rma_register(ptrs,n)    tell a one-sided implementation which arrays get exchanged
//   C.t_halo, C.t_allreduce   accumulated communication time (filled by the policy)
```

Any object that provides these can be plugged in. The serial driver plugs in an object whose `exchange` does nothing and whose `sum` returns its argument unchanged (D7). The MPI driver plugs in one that calls MPI. In C++ this is done with a template: `template <class Comm> Result solve(const Params& P, const Local& L, Comm& C, const char* impl)`.

The first helper refreshes halos and applies a kernel, hiding communication when allowed:

```cpp
template <class Comm, class Kernel>
inline void exchange_and_apply(Comm& C, const Local& L, double* u, Kernel kernel, double& t_compute) {
    if (C.can_overlap() && has_deep_interior(L)) {
        C.start(u);
        double t0 = C.wtime();
        kernel(deep_interior(L));
        t_compute += C.wtime() - t0;
        C.wait();
        t0 = C.wtime();
        for_each_boundary_slab(L, kernel);
        t_compute += C.wtime() - t0;
    } else {
        C.exchange(u);
        double t0 = C.wtime();
        kernel(full(L));
        t_compute += C.wtime() - t0;
    }
}
```

In overlap mode: post the messages, compute the inside while they travel, wait for them, then compute the outer layer that needed the fresh halo values. Otherwise: exchange, then compute everything. Only the kernel time is added to `t_compute`. The `Comm` object adds its own time to `C.t_halo` inside `start`, `wait` and `exchange`.

The solver allocates three arrays of `L.ntot` doubles (`u`, `un`, `f`), fills `f`, and computes `norm(f)` once, timing that as setup. For Jacobi:

```cpp
        double* cur = u.data();
        double* nxt = un.data();
        double* bufs[2] = {cur, nxt};
        C.rma_register(bufs, 2);
        for (it = 1; it <= P.iters; ++it) {
            exchange_and_apply(C, L, cur,
                [&](Range r) { sweep(cur, nxt, r); }, t_compute);
            std::swap(cur, nxt);
```

`cur` points at the current iterate and `nxt` at the buffer being written. The expression `[&](Range r) { ... }` is a **lambda**: a small unnamed function written in place that can see the surrounding variables (that is what `[&]` means), like a sticky note that says "when asked, do this with those ingredients on the counter". It is what gets passed as `kernel`. After the sweep the two pointers are swapped, so no data is copied, and the new iterate simply becomes `cur`. `rma_register` is a no-op except in the one-sided exchange mode, where both buffers must be exposed as windows in advance (D6, D7).

```cpp
            if (check_every > 0 && it % check_every == 0) {
                // residual needs fresh halos of the new iterate
                C.exchange(cur);
                double tt = C.wtime();
                double local = residual(L, cur, f.data(), nxt);   // nxt is scratch here
                t_compute += C.wtime() - tt;
                rel = std::sqrt(C.sum(local)) / fnorm;
                if (P.verbose && C.rank() == 0) std::printf("  it %ld  rel_resid %.3e\n", it, rel);
                if (P.tol > 0 && rel < P.tol) { converged = 1; break; }
            }
```

Checking convergence is not free: it needs an extra halo exchange, a full pass to compute the residual (written into `nxt`, which is free at this point), and a global sum. That is why `check_every` exists and why it defaults to 10 when a tolerance is given (`if (P.solver == "jacobi" && P.tol > 0 && check_every == 0) check_every = 10;`) and to "never" when running a fixed number of sweeps. After the loop, `if (it > P.iters) it = P.iters;` repairs the counter, which overshoots by one when the loop runs to the end, and if the final iterate lives in `un` it is copied into `u`.

Conjugate gradient needs three more arrays, `r`, `p` and `Ap`:

```cpp
        std::copy(f.begin(), f.end(), r.begin());
        std::copy(f.begin(), f.end(), p.begin());
        double* pb[1] = {p.data()};
        C.rma_register(pb, 1);
        double rr = C.sum(dot(L, r.data(), r.data()));
        rel = std::sqrt(rr) / fnorm;
        for (it = 1; it <= P.iters; ++it) {
            exchange_and_apply(C, L, p.data(),
                [&](Range rg) { applyA(p.data(), Ap.data(), rg); }, t_compute);
            double tt = C.wtime();
            double pAp_local = dot(L, p.data(), Ap.data());
            t_compute += C.wtime() - tt;
            const double pAp = C.sum(pAp_local);
            const double alpha = rr / pAp;
            tt = C.wtime();
            axpy(L,  alpha, p.data(),  u.data());
            axpy(L, -alpha, Ap.data(), r.data());
            double rr_local = dot(L, r.data(), r.data());
            t_compute += C.wtime() - tt;
            const double rr_new = C.sum(rr_local);
            rel = std::sqrt(rr_new) / fnorm;
            if (P.tol > 0 && rel < P.tol) { converged = 1; break; }
            tt = C.wtime();
            xpay(L, r.data(), rr_new / rr, p.data());
            t_compute += C.wtime() - tt;
            rr = rr_new;
        }
```

Because the initial guess is zero, the initial residual `r = f - A 0` is just `f`, and the first search direction `p` is `r`. Each iteration refreshes the halos of `p` and computes `Ap`, forms the global dot product `pAp` (one all-reduce), computes the step length `alpha = rr / pAp`, moves the solution along `p` and updates the residual, forms the new `rr` (second all-reduce), which gives the relative residual for free, and finally updates the direction with `beta = rr_new / rr`. Only `p` is ever exchanged, because `Ap` is the only stencil application. Two global sums per iteration is the communication price of CG that Part B discussed.

Timers: `t_setup` wraps `init_problem` and the first norm. `t_total` runs from a barrier just before the loop to a barrier just after, so it is the time of the slowest process. `t_compute` is accumulated around every kernel call. `C.t_halo` and `C.t_allreduce` are accumulated inside the Comm object. After the loop the code refreshes the halos once more, computes the true residual (`f - A u`, not the recursively updated one CG carries) and the two error norms, scales the squared error by the cell volume `hx * hy * hz` to get the discrete L2 norm, and fills every field of `Result`, taking the maximum and the mean of each timer over ranks via `C.max` and `C.sum(...) / C.size()`. These diagnostics are deliberately outside the timed region.

**What came out of it?** The same `solve` function, compiled five ways, produced identical results in the validation runs: for the fixed 300-sweep Jacobi case on a 300 by 300 grid, every binary reports `rel_resid 9.707e-01, l2_err 8.370888e-01, linf_err 1.703428e+00`. That is the expected outcome of writing the loop once.

**How would I explain it to a classmate in two sentences?** The solver is a template that takes a "communicator" object as a plug-in, so the serial and the MPI programs run literally the same loop. Jacobi swaps two buffers each sweep and pays for a convergence check only every so often, while CG does one stencil application and two global sums per iteration.

One more thing the solver does since the data extension: `solve` takes an optional pointer to a `DataBlock` (D4a). When it is given, the three stencil kernels are the masked versions and the right-hand side comes from `finish_data`; when it is null, the manufactured problem is set up as before. The choice is made once through three small lambda functions at the top of `solve`, so the Jacobi and CG loops themselves do not contain a single `if` about which problem they are solving. After the loop, the data path computes the hidden-pixel errors and hands the solved anomaly back through `DataBlock::solution` so the driver can write the filled field, and the manufactured path computes the error against the exact formula as before. One guard was added for the data: if the right-hand side is exactly zero (nothing to fill) or the CG curvature `pAp` is exactly zero (nothing left to solve), the loop stops instead of dividing by zero.

## D6. src/decomp.hpp: cutting the grid between processes and swapping halos

**What is it?** `decomp.hpp` is the only file that knows how the global grid is divided among MPI processes and how neighbouring processes exchange their boundary layers. MPI (Message Passing Interface) is the library that lets separate processes, possibly on different computers, send each other messages. Each process is called a **rank** and is identified by a number, like a numbered desk in an office where staff pass notes. A **communicator** is a named group of ranks within which messages are addressed, like one department's internal mail.

**Why did we do it?** Domain decomposition with halo exchange is the standard pattern for stencil codes (Lecture 12), and the design question the report studies is how to do the exchange. Isolating the four strategies in one header makes them directly comparable, since nothing else in the program changes between them.

**How does it work?** The state is a struct:

```cpp
struct Decomp {
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0, size = 1;
    int dims[3] = {1, 1, 1}, coords[3] = {0, 0, 0};
    int lo[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};  // neighbour towards smaller index
    int hi[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};  // neighbour towards larger index
    Local L;
    MPI_Datatype send_lo[3], recv_lo[3], send_hi[3], recv_hi[3];
    MPI_Datatype nbr_lo_halo[3], nbr_hi_halo[3];
    MPI_Win win = MPI_WIN_NULL;
    int ndims_active = 3;   // 2 for a 2D problem (no z exchange)
};
```

`cart` is a communicator in which ranks are arranged as a grid (a Cartesian communicator), `dims` is the shape of that grid (for example 4 by 4 by 1), `coords` is this rank's position in it, and `lo[d]` and `hi[d]` are the ranks of the neighbours in each direction. `MPI_PROC_NULL` is a special value meaning "no neighbour there": sends to it and receives from it are silently skipped, which is exactly right at the physical boundary where the halo must stay zero. `L` is this rank's block geometry (D3). The `MPI_Datatype` fields describe the shape of a face in memory, and `win` is used by the one-sided mode.

A **datatype** in MPI is a description of which bytes of a buffer to pick out. A face of the block is not contiguous in memory (the left face is one cell from each row, with a stride between them), so instead of copying the face into a temporary buffer, the code describes it:

```cpp
inline MPI_Datatype face_type(int nx, int ny, int nz, int d, int pos) {
    int sizes[3]    = {nz + 2, ny + 2, nx + 2};  // C order: slowest first (k, j, i)
    int subsizes[3] = {nz, ny, nx};
    int starts[3]   = {1, 1, 1};
    int cd = 2 - d;                               // dimension d=0 (x) is the last C index
    subsizes[cd] = 1; starts[cd] = pos;
    MPI_Datatype t;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &t);
    MPI_Type_commit(&t);
    return t;
}
```

`MPI_Type_create_subarray` describes a rectangular sub-box of a bigger box. `sizes` is the full block including halos, written slowest-varying first (k, then j, then i), which is what `MPI_ORDER_C` means. `subsizes` starts as the whole interior and `starts` at (1,1,1), then the chosen dimension `d` is squeezed to thickness 1 at plane `pos`. Because the arrays are in k-j-i order, dimension x (`d = 0`) is index 2, hence `cd = 2 - d`. Calling with `pos = 1` selects the first interior plane (the data to send to the low neighbour), `pos = 0` the low halo plane (where data from the low neighbour lands), `pos = n_d` the last interior plane and `pos = n_d + 1` the high halo plane. Committing the type lets MPI use it.

`make_decomp` builds everything in order:

```cpp
    MPI_Dims_create(D.size, 3, D.dims);            // fills in the zeros as evenly as possible
    int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, D.dims, periods, 1, &D.cart);
    MPI_Comm_rank(D.cart, &D.rank);
    MPI_Cart_coords(D.cart, D.rank, 3, D.coords);
    for (int d = 0; d < 3; ++d) MPI_Cart_shift(D.cart, d, 1, &D.lo[d], &D.hi[d]);
```

First the process-grid shape: any zeros in `dims` (the default) are filled by `MPI_Dims_create` as evenly as possible, and for a 2D problem `dims[2]` was forced to 1 beforehand so the grid is flat. Then a Cartesian communicator is created with no wrap-around (`periods` all zero) and permission to reorder ranks (`1`) so MPI can place neighbours close together. `MPI_Cart_shift` returns the neighbour one step down and one step up in each direction, or `MPI_PROC_NULL` at the edge.

```cpp
    if (P.local_n > 0) for (int d = 0; d < 3; ++d) P.n[d] = P.local_n * D.dims[d];
    if (P.dim == 2) P.n[2] = 1;
    ...
    int cnt[3]; long off[3];
    for (int d = 0; d < 3; ++d) partition(P.n[d], D.dims[d], D.coords[d], cnt[d], off[d]);
    D.L = make_local(P, P.n, cnt[0], cnt[1], cnt[2], off[0], off[1], off[2]);
```

Weak scaling is implemented right here: if `--local-n` was given, the global size in each direction becomes `local_n` times the number of ranks in that direction, so each rank owns exactly `local_n` points per side whatever the rank count. Then each direction is cut with `partition` (D3) and the local geometry is built. The code aborts with a message if the grid is smaller than the process grid in any direction.

The datatypes come next. Besides the four for this rank's own faces, the code builds types describing the neighbour's halo plane, which the one-sided mode needs:

```cpp
        int ncnt[3] = {nx, ny, nz}; long noff;
        if (D.lo[d] != MPI_PROC_NULL) {
            partition(P.n[d], D.dims[d], D.coords[d] - 1, ncnt[d], noff);
            D.nbr_lo_halo[d] = face_type(ncnt[0], ncnt[1], ncnt[2], d, ncnt[d] + 1); // their hi halo
        } else D.nbr_lo_halo[d] = MPI_DATATYPE_NULL;
```

The neighbour's block can be one point larger or smaller than ours along the split direction (recall `partition` gives the first few ranks an extra point), so its memory layout differs. The code recomputes the neighbour's size with `partition` at coordinate minus one and describes the high halo plane of a block of that size. With a one-sided put (below), the sender writes directly into the receiver's memory, so the sender must describe the receiver's layout exactly.

The four strategies. Blocking, one direction at a time:

```cpp
inline void exchange_sendrecv(const Decomp& D, double* u) {
    for (int d = 0; d < D.ndims_active; ++d) {
        MPI_Sendrecv(u, 1, D.send_hi[d], D.hi[d], 10 + d,
                     u, 1, D.recv_lo[d], D.lo[d], 10 + d, D.cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(u, 1, D.send_lo[d], D.lo[d], 20 + d,
                     u, 1, D.recv_hi[d], D.hi[d], 20 + d, D.cart, MPI_STATUS_IGNORE);
    }
}
```

`MPI_Sendrecv` sends one thing and receives another in a single call, which cannot deadlock. The first call sends my high face to the high neighbour and receives the low neighbour's high face into my low halo, and the second does the reverse. A **tag** is a small integer label on a message so the receiver can tell which message is which, like a subject line. Tags `10 + d` and `20 + d` distinguish the two directions of travel in each dimension. Only `ndims_active` dimensions are exchanged, so a 2D problem never touches z.

Non-blocking:

```cpp
inline int exchange_start(const Decomp& D, double* u, MPI_Request* req) {
    int n = 0;
    for (int d = 0; d < D.ndims_active; ++d) {
        MPI_Irecv(u, 1, D.recv_lo[d], D.lo[d], 10 + d, D.cart, &req[n++]);
        MPI_Irecv(u, 1, D.recv_hi[d], D.hi[d], 20 + d, D.cart, &req[n++]);
    }
    for (int d = 0; d < D.ndims_active; ++d) {
        MPI_Isend(u, 1, D.send_hi[d], D.hi[d], 10 + d, D.cart, &req[n++]);
        MPI_Isend(u, 1, D.send_lo[d], D.lo[d], 20 + d, D.cart, &req[n++]);
    }
    return n;
}
inline void exchange_wait(int n, MPI_Request* req) { MPI_Waitall(n, req, MPI_STATUSES_IGNORE); }
```

`MPI_Irecv` and `MPI_Isend` return immediately and hand back a request handle. All receives are posted first (so the data has somewhere to land), then all sends, and `MPI_Waitall` blocks until every one has completed. The "isend" strategy calls `exchange_start` then `exchange_wait` back to back; the "overlap" strategy puts the deep-interior computation between them (D5). With up to 3 dimensions and 4 requests each, 12 request slots suffice.

One-sided:

```cpp
inline void exchange_rma(Decomp& D, double* u) {
    MPI_Win_fence(MPI_MODE_NOPRECEDE, D.win);
    for (int d = 0; d < D.ndims_active; ++d) {
        if (D.hi[d] != MPI_PROC_NULL) MPI_Put(u, 1, D.send_hi[d], D.hi[d], 0, 1, D.nbr_hi_halo[d], D.win);
        if (D.lo[d] != MPI_PROC_NULL) MPI_Put(u, 1, D.send_lo[d], D.lo[d], 0, 1, D.nbr_lo_halo[d], D.win);
    }
    MPI_Win_fence(MPI_MODE_NOSUCCEED, D.win);
}
```

A **window** is a region of a rank's memory that it has agreed to let other ranks read or write directly (Lecture 8's Remote Memory Access). `MPI_Win_fence` opens and closes an "epoch" during which puts are allowed. The two `MPI_MODE` hints tell MPI that nothing precedes the first fence and nothing follows the second, which lets it skip some synchronisation. Inside, each rank writes its high face into the high neighbour's low halo and its low face into the low neighbour's high halo. The target is described by displacement 0 plus the neighbour-halo datatype built earlier, which is why that datatype had to exist. `rma_attach` creates a window over an array; the MPI driver manages windows for the arrays the solver registers (D7). `free_decomp` frees every datatype, the window if any, and the communicator. `dims_string` formats `dims` as text like `4x4x1` for the CSV.

**What came out of it?** All four strategies produce identical results in the validation table, which confirms that the datatypes and tags are consistent (a wrong `starts` entry or a swapped tag would corrupt a halo and the error would differ). Which strategy is fastest is an empirical question answered in Part E.

**How would I explain it to a classmate in two sentences?** The grid is cut into a Cartesian grid of blocks, each rank finds its neighbours with `MPI_Cart_shift`, and faces are described rather than copied using subarray datatypes. The four exchange functions differ only in the MPI calls they make: blocking send and receive, nonblocking with a wait, nonblocking with computation in between, or one-sided puts between fences.

## D7. src/serial_main.cpp and src/mpi_main.cpp: the two drivers

**What is it?** The drivers are the files with `main`, the function where a program starts. Each parses the options, builds the geometry, constructs a `Comm` object, calls `solve`, prints and records the result. `serial_main.cpp` becomes `poisson_serial` or `poisson_omp`, and `mpi_main.cpp` becomes `poisson_mpi` or `poisson_mpi_omp`.

**Why did we do it?** The assignment asks for a genuine serial implementation as the baseline, not an MPI program run on one rank. `serial_main.cpp` never includes `mpi.h`, so the baseline has no MPI overhead at all. The MPI driver is where MPI-specific resource handling (thread mode, windows) belongs, away from the solver logic.

**How does it work?** The serial Comm is almost all no-ops:

```cpp
struct SerialComm {
    double t_halo = 0.0, t_allreduce = 0.0;
    double wtime() const { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    void exchange(double*) {}
    bool can_overlap() const { return false; }
    void start(double*) {}
    void wait() {}
    void rma_register(double**, int) {}
    double sum(double x) const { return x; }
    double max(double x) const { return x; }
    void barrier() const {}
    int rank() const { return 0; }
    int size() const { return 1; }
```

`exchange` does nothing because there are no neighbours, and `sum` of one process's number is that number. The clock is `std::chrono::steady_clock`, the same timer the lecturer's sample code uses. `threads()` returns `omp_get_max_threads()` when compiled with OpenMP and 1 otherwise, and `main` sets the implementation name the same way:

```cpp
#ifdef _OPENMP
    if (P.threads > 0) omp_set_num_threads(P.threads);
    const char* impl = "omp";
#else
    const char* impl = "serial";
#endif
```

`#ifdef _OPENMP` is a compile-time switch: the lines between are kept only when OpenMP is on. `--local-n` in the serial driver simply sets `n` to that value, so a one-rank weak-scaling point can be produced by the serial binary too. Then `make_local` is called with the whole grid as one block at offset zero, `solve` runs, and the result is printed and appended to the CSV.

The MPI Comm wraps a `Decomp` and the exchange mode:

```cpp
    void exchange(double* u) {
        const double t0 = MPI_Wtime();
        if (mode == "sendrecv") exchange_sendrecv(D, u);
        else if (mode == "rma") {
            MPI_Win w = MPI_WIN_NULL;
            for (auto& p : wins) if (p.first == u) w = p.second;
            if (w == MPI_WIN_NULL) {               // array not registered: fall back to messages
                nreq = exchange_start(D, u, req); exchange_wait(nreq, req);
            } else { D.win = w; exchange_rma(D, u); D.win = MPI_WIN_NULL; }
        } else { nreq = exchange_start(D, u, req); exchange_wait(nreq, req); }
        t_halo += MPI_Wtime() - t0;
    }
```

Every exchange is timed with `MPI_Wtime` and added to `t_halo`. In RMA mode the object looks up the window belonging to the array being exchanged. Windows are created when the solver calls `rma_register`:

```cpp
    void rma_register(double** bufs, int n) {
        if (mode != "rma") return;
        for (int i = 0; i < n; ++i) {
            MPI_Win w;
            MPI_Win_create(bufs[i], (MPI_Aint)(D.L.ntot * sizeof(double)), sizeof(double), MPI_INFO_NULL, D.cart, &w);
            wins.emplace_back(bufs[i], w);
        }
    }
```

Jacobi registers both of its buffers (since the exchanged array alternates every sweep) and CG registers `p`. The destructor `~MpiComm` frees the windows. `sum` and `max` call `MPI_Allreduce` with `MPI_SUM` or `MPI_MAX` and add the elapsed time to `t_allreduce`. `barrier` calls `MPI_Barrier` on the Cartesian communicator. `start` and `wait` wrap `exchange_start` and `exchange_wait` with timing for the overlap mode.

`main` initialises MPI with `MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided)`. `FUNNELED` tells MPI that the program is multi-threaded but only the main thread will ever call MPI, which is true here: OpenMP threads exist only inside the kernels, and every MPI call happens between kernels on the main thread. This is the cheapest thread-safety level that is still correct. The driver then builds the decomposition, constructs the Comm inside a block `{ ... }` so that its destructor runs (freeing windows) before `free_decomp` and `MPI_Finalize`, and only rank 0 prints and writes the CSV row, so each run yields exactly one row.

**What came out of it?** `poisson_serial` and `poisson_mpi` on one rank solve the same problem. The difference between their times is the MPI overhead (barriers, the degenerate exchanges to `MPI_PROC_NULL`, the all-reduces of a single value), which Part E reports.

**How would I explain it to a classmate in two sentences?** The serial driver plugs a do-nothing communicator into the solver, and the MPI driver plugs in one that times every MPI call. The MPI driver also manages the one-sided windows and asks MPI for the funneled thread mode because only the main thread talks to MPI.

## D8. src/cuda: the GPU version and how it was tested without a GPU

**What is it?** Three files implement MPI+CUDA. `gpu_kernels.cuh` holds the kernels, `mpi_cuda_main.cu` is the driver, and `cuda_compat.hpp` is a shim that lets the same two files compile and run as ordinary C++ on a machine with no GPU. CUDA is NVIDIA's language extension for writing code that runs on their graphics processors. A CUDA **kernel** is a function that the GPU runs simultaneously in thousands of copies, one per thread, like handing every student in a lecture hall the same worksheet with a different row number.

**Why did we do it?** The rubric awards 10 points for the MPI+X part and states that choosing OpenMP as X also requires a GPU test. The laptop the code was written on has no NVIDIA GPU and no `nvcc`, and the GPU nodes of Vanda are only reachable through the queue. Without the shim, the GPU code could not have been run at all before submission. With it, the logic has been checked against the CPU versions. To be plain about the status: the CUDA code has been compiled and run only through this emulation. It has never been compiled with `nvcc` or executed on a real GPU, so its performance is unknown and compile errors specific to `nvcc` are possible.

**How does it work?** The shim first:

```cpp
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define KERNEL_LAUNCH(kernel, grid, block, ...) kernel<<<grid, block>>>(__VA_ARGS__)
#define POISSON_CUDA_REAL 1
#else
#define POISSON_CUDA_REAL 0
#define __global__
#define __device__
#define __host__
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
struct uint3 { unsigned x, y, z; };
inline uint3 threadIdx, blockIdx;
inline dim3 blockDim, gridDim;
```

`__CUDACC__` is defined only when `nvcc` is compiling. In that case `KERNEL_LAUNCH(k, grid, block, args)` expands to the normal CUDA launch `k<<<grid, block>>>(args)`, where the triple angle brackets say how many blocks of how many threads to start. Otherwise the CUDA keywords `__global__`, `__device__` and `__host__` are defined as nothing, so a kernel becomes an ordinary function, and the built-in thread coordinates `threadIdx`, `blockIdx`, `blockDim`, `gridDim` become global variables. Then:

```cpp
template <class Fn> inline void emul_launch(dim3 grid, dim3 block, Fn fn) {
    gridDim = grid; blockDim = block;
    for (unsigned bz = 0; bz < grid.z; ++bz) for (unsigned by = 0; by < grid.y; ++by) for (unsigned bx = 0; bx < grid.x; ++bx) {
        blockIdx = {bx, by, bz};
        for (unsigned tz = 0; tz < block.z; ++tz) for (unsigned ty = 0; ty < block.y; ++ty) for (unsigned tx = 0; tx < block.x; ++tx) {
            threadIdx = {tx, ty, tz};
            fn();
        }
    }
}
#define KERNEL_LAUNCH(kernel, grid, block, ...) emul_launch(grid, block, [&]() { kernel(__VA_ARGS__); })
```

The emulated launch loops over every block and every thread in sequence, sets the coordinate globals, and calls the kernel. A real GPU runs those calls in parallel. The emulation runs them one after another, which is slow but gives the same arithmetic. `cudaMalloc` becomes `malloc`, `cudaMemcpy` becomes `memcpy`, `cudaDeviceSynchronize` does nothing, and `atomicAdd` on a double is a plain addition (on a real GPU it is an uninterruptible add that several threads can use safely). The `CUDA_CHECK` macro wraps every CUDA call and aborts with the file and line if it fails.

The kernels file defines a plain-data copy of the geometry (`Geo`) and a box (`Box`), because the `Local` struct's member function cannot be called from GPU code as written. The Jacobi kernel:

```cpp
__global__ void k_jacobi(Geo g, Box b, const double* u, double* un, const double* f) {
    int i = b.i0 + blockIdx.x * blockDim.x + threadIdx.x;
    int j = b.j0 + blockIdx.y * blockDim.y + threadIdx.y;
    int k = b.k0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= b.i1 || j >= b.j1 || k >= b.k1) return;
    size_t c = cell(g, i, j, k);
    un[c] = (f[c] + g.ihx2 * (u[c - 1] + u[c + 1]) + g.ihy2 * (u[c - g.sy] + u[c + g.sy])
                  + g.ihz2 * (u[c - g.sz] + u[c + g.sz])) / g.diag;
}
```

Instead of loops, each thread computes which cell it owns from its block and thread coordinates, returns if that cell is outside the box (the last block in each direction is usually only partly inside), and performs the same arithmetic as the CPU sweep. Threads are laid out with `i` fastest (`box_block()` returns 64 by 4 by 1 threads), so consecutive threads touch consecutive memory addresses, which GPUs need for speed (coalesced access, Lecture 10 and Lecture 11). `box_grid` divides the box size by the block size, rounding up. `k_applyA`, `k_axpy` and `k_xpay` follow the same pattern.

Reductions use a different pattern:

```cpp
__global__ void k_dot(Geo g, const double* a, const double* b, double* out) {
    const long n = (long)g.nx * g.ny * g.nz;
    const long stride = (long)gridDim.x * blockDim.x;
    double s = 0.0;
    for (long t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        int i = 1 + (int)(t % g.nx);
        long rest = t / g.nx;
        int j = 1 + (int)(rest % g.ny);
        int k = 1 + (int)(rest / g.ny);
        size_t c = cell(g, i, j, k);
        s += a[c] * b[c];
    }
    atomicAdd(out, s);
}
```

A fixed number of threads (`REDUCE_BLOCKS = 1024` times `REDUCE_THREADS = 256`, about 262 thousand) each walk through the interior with a stride equal to the total thread count, accumulate a private sum, and add it once to a single device variable with `atomicAdd`. This is a "grid-stride loop", simple and free of shared memory, at the cost of 262 thousand atomic additions per dot product. `k_residual` is the same with the stencil inside. The face kernels `k_pack` and `k_unpack` copy one plane of the block into or out of a contiguous buffer. `face_cell` maps a flat index to the cell on the face for each of the three dimensions, and `face_size` gives the plane's area.

The driver keeps per-direction, per-side buffers on both the GPU and the host:

```cpp
struct GpuHalo {
    double *d_send[3][2] = {}, *d_recv[3][2] = {}, *h_send[3][2] = {}, *h_recv[3][2] = {};
    long size[3] = {0, 0, 0};
    MPI_Request req[12]; int nreq = 0;
};
```

`d_` buffers live in device memory and `h_` buffers in **pinned memory**, host memory that the operating system promises not to move, so the GPU can copy to it directly (`cudaMallocHost`). `halo_start` launches pack kernels for every face that has a neighbour, synchronises, copies the packed faces to the host unless `--cuda-aware` was given, and posts the receives and sends with the same tags as the CPU code. `halo_finish` waits, copies the received faces to the device and launches unpack kernels into the halo planes. With `--cuda-aware`, device pointers are handed to MPI directly and both copies are skipped, which only works if the MPI library was built with CUDA support (the lecturer's Lecture 10 slide on CUDA-aware MPI).

`gpu_dot` zeroes the device scalar, launches `k_dot`, copies the single result back and does one `MPI_Allreduce`. `exchange_and_apply` mirrors the CPU version: in overlap mode it starts the exchange, launches the deep-interior kernel (which on a real GPU runs asynchronously while the messages travel), finishes the exchange, then launches the boundary slabs. The Jacobi and CG loops are near-copies of `solver.hpp` with kernel launches in place of function calls and `cudaDeviceSynchronize` before every timer read. One caveat on the timers in overlap mode: because a launch returns before the kernel finishes, the deep-interior kernel's execution time is counted inside `halo_finish` (which synchronises) and appears under `t_halo` rather than `t_compute`. The total is right, but the split between the two is approximate in that one mode.

Device selection uses `MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, ...)` to learn this rank's position among the ranks sharing the node, then `cudaSetDevice(node_rank % ndev)`, so that two ranks on a two-GPU node each take a different GPU. At the end the solution is copied back to the host and the error against the exact solution is computed with the CPU `error_norms`, and the true residual is computed on the device. The implementation name written to the CSV is `mpi_cuda` under `nvcc` and `mpi_cuda_emul` under the shim, so the two can never be confused in the results.

**What came out of it?** Under emulation, with 3 to 8 ranks, in 2D and 3D, with and without overlap and with the `--cuda-aware` path, the CUDA driver reproduced the serial results to every printed digit for fixed-sweep Jacobi and to the last digit of the error for CG. That validates the kernels, the face packing, the tags and the solver logic. It says nothing about speed on a GPU.

**How would I explain it to a classmate in two sentences?** The GPU code replaces each CPU loop with a kernel where every thread handles one cell, and swaps halos by packing faces into buffers, copying them to the CPU and sending them with MPI. A sixty-line shim lets the same files compile as normal C++ so the logic was tested on a laptop, but the real GPU build has not yet been compiled or timed.

## D8a. tools/gridtool.cpp, scripts/data/download.sh and scripts/local/maps.sh: getting and looking at the data

**What is it?** Three small programs that stand between NOAA's server and the solver. `scripts/data/download.sh` fetches the daily maps. `bin/gridtool` (built from `tools/gridtool.cpp`, about 300 lines of C++) converts them into the grid file, resamples the reference, prints statistics and exports single days as text matrices. `scripts/local/maps.sh` uses those exports and `analysis/maps.gp` to draw the before-and-after maps.

**Why did we do it?** The rule of the project is no Python, so the usual data-wrangling route (a notebook with a NetCDF library) was closed. ERDDAP's plain-text output and a few hundred lines of C++ turned out to be enough, and they have the advantage that every step is a command that can be re-run and inspected.

**How does it work?** `download.sh REGION START [END] [ref]` builds one URL per day. The URL names the dataset, the variable and the ranges in square brackets, for example `sea_surface_temperature[(2025-04-10T12:00:00Z)][(0.0)][(5.0):(25.0)][(65.0):(95.0)]`, and asks for the `.asc` format, which is a human-readable grid: a header describing the array, then one line per latitude row with the values separated by commas (an empty value is a missing pixel), then the latitude and longitude axes. `curl -g` fetches it (the `-g` stops curl from interpreting the square brackets), with three attempts and a resume-friendly rule that skips files already present. The region `india` is the box 5 N to 25 N, 65 E to 95 E; `singapore` is 8 S to 8 N, 96 E to 120 E; any `S,N,W,E` works. A year of the India box is 365 files of about 1 MB plus, with `ref`, 365 reference files of about 2 MB.

`gridtool convert OUT.grid --mask MASK.asc DAY.asc ...` reads every daily file with a small parser (`read_asc`), sorts the days by file name (which ends in the date), reorders the rows so latitude ascends, resamples the land mask from its 5 km grid to the 4 km grid by nearest neighbour (the blended product's mask is a bit field in which bit 2 means land), writes the header and then the slabs one by one, printing the observed share of sea pixels for each day. `gridtool regrid TARGET.grid OUT.grid --offset -273.15 REF.asc ...` does the same for the reference, matching each target pixel to the nearest reference pixel and converting kelvin to Celsius. `gridtool info FILE.grid [--days]` prints the header and the counts of observed and missing sea pixels and of land, overall or per day. `gridtool slice FILE.grid K OUT.txt` writes day K as a text matrix with `NaN` where there is nothing to draw, and `gridtool diff A.grid KA B.grid KB OUT.txt` writes day KA of one grid minus day KB of another and prints their root-mean-square difference. `maps.sh GRID K FILL2D FILL3D OUT.png` slices the observed day, the 2D fill and the day K of the 3D fill, computes the 3D-minus-2D difference restricted to the filled pixels (an `awk` line blanks the observed ones), reads the pixel coordinates from the first line of the slice, and calls gnuplot on `analysis/maps.gp`, which draws four maps with a common colour scale from 24 to 32 degrees and a diverging scale from -2 to +2 for the difference.

**What came out of it?** `data/india_2025.grid` (801 x 534 x 365, 625 MB), `data/india_2025_ref.grid`, the per-day statistics quoted in C1 and the map figure `results/figures/sst_maps.png` discussed in E17.

**How would I explain it to a classmate in two sentences?** A shell script downloads one text grid per day from NOAA's ERDDAP server, and a C++ tool stacks them into a binary file the solver can read block by block. The same tool exports single days as text so gnuplot can draw the observed and filled maps.

## D9. scripts/local/run_all.sh: every experiment on one machine

**What is it?** A bash shell script that runs all the local experiments in a fixed order and writes one CSV per experiment family into `results/local/`, plus a `run.log` with timestamps. It takes two optional arguments: the number of cores to use (default 16) and the number of repeats per timing (default 2).

**Why did we do it?** The report needs the same experiment families on the laptop (for development and for figures that do not need a cluster) and on the cluster. Scripting them means a figure can be regenerated with one command and the exact grid sizes and rank counts are written down rather than remembered.

**How does it work?** After checking that the four CPU binaries exist, it builds the list of rank counts as powers of two up to `MAXCORES` (`RANKS=(1 2 4 8 16)` for 16 cores) and defines four helpers:

```bash
run_serial() { # csv tag args...
  local csv=$1 tag=$2; shift 2
  for ((k=0;k<REPEATS;k++)); do $S --csv "$csv" --tag "$tag" "$@" | grep -E '^iters=|^t_total' | tr '\n' ' ' | tee -a "$LOG"; echo | tee -a "$LOG"; done
}
run_mpi() { # csv tag np args...
  local csv=$1 tag=$2 np=$3; shift 3
  for ((k=0;k<REPEATS;k++)); do mpirun -np $np $M --csv "$csv" --tag "$tag" "$@" | grep -E '^t_total' | tee -a "$LOG"; done
}
run_hybrid() { # csv tag np threads args...
  local csv=$1 tag=$2 np=$3 th=$4; shift 4
  for ((k=0;k<REPEATS;k++)); do OMP_NUM_THREADS=$th mpirun -np $np $MO --threads $th --csv "$csv" --tag "$tag" "$@" | grep -E '^t_total' | tee -a "$LOG"; done
}
```

Each helper takes the CSV path and a tag, then passes the remaining arguments (`"$@"`) straight to the binary, repeating `REPEATS` times. `OMP_NUM_THREADS` is an **environment variable**, a named setting the operating system hands to every program it starts, like a note left on the desk before the shift begins. OpenMP reads it to decide how many threads to use. The `tee -a` sends one summary line to the terminal and to the log. The experiment blocks and the CSVs they write:

| block | CSV | tags | what varies |
|---|---|---|---|
| 1 validation | validation.csv | order2d, order3d, jacobi_conv, agree | grid size at tight tolerance; serial vs every variant on the same problem; the emulated CUDA driver if present |
| 2 serial scaling | serial_scaling.csv | jac2d, jac3d, cg2d, cg3d | 2D n = 128 to 4096 and 3D n = 16 to 256 for Jacobi (200 sweeps); 2D n = 128 to 2048 and 3D n = 32 to 256 for CG to 1e-8 |
| 3 strong scaling | strong.csv | the same four | fixed problems (2D 2048, 3D 192 Jacobi; 2D 1024, 3D 128 CG) at 1, 2, 4, 8, 16 ranks, plus the serial baseline |
| 4 weak scaling | weak.csv | the same four | `--local-n` 1024 (2D) and 64 (3D) for Jacobi, 512 and 48 for CG with a fixed 300 iterations, at every rank count |
| 5 design | exchange.csv, decomp.csv, checkevery.csv | jac2d, jac3d, cg2d | the four exchange modes at 8 and 16 ranks; five 2D and six 3D process-grid shapes at 16 ranks; residual check every 0, 1, 10, 100 sweeps |
| 6 hybrid | hybrid.csv | the same four | ranks x threads = 16x1, 8x2, 4x4, 2x8, 1x16 |
| 7 OpenMP only | omp_threads.csv | the same four | 1, 2, 4, 8, 16 threads with `poisson_omp` |

The grid sizes were chosen so that the largest Jacobi problems (4096 squared and 256 cubed, both about 16.8 million points) take seconds rather than minutes per 200 sweeps on a laptop, and so that the strong-scaling problems leave at least 262 thousand points per rank at 16 ranks (2048 squared divided by 16), enough that computation still dominates. Weak scaling for CG uses a fixed iteration count because the number of CG iterations to a tolerance grows with the global grid, which would otherwise mix two effects. Each block deletes its CSV first (`rm -f "$CSV"`) so a rerun starts clean.

**What came out of it?** On this laptop the script was started with `scripts/local/run_all.sh 16 2`. Then the design, hybrid and OpenMP stages were run again after a Bash 3 bug fix, and the strong and weak stages again with `EXTRA_RANKS=12`. Every CSV listed in D1 exists. The measured wall-clock time of each block is in `results/local/run.log` and summarised in D13.

**How would I explain it to a classmate in two sentences?** One script runs every experiment twice and records a CSV row per run, labelled with a tag that the analysis script uses to group them. Change `MAXCORES` and it adapts the rank list to your machine.

Stage 8, added with the data extension and selected by the stage name `data`, runs only if the grid file exists (`GRID`, default `data/india_2025.grid`; `DAY`, default 99, is the 2D day; `DAYS3D` can shorten the 3D stack for a quick test). It runs the serial baselines (CG to a relative residual of 1e-6 and Jacobi for a fixed 1000 sweeps on the day, CG to 1e-6 and Jacobi for 200 sweeps on the year) writing the filled fields, then the same four cases on every rank count, then OpenMP alone and one MPI+OpenMP split at the full core count, then the accuracy study at the full core count (patch sides 1, 4, 16 and 64 in 2D and 3D, the 3D error both over the year and on the 2D day alone, and time weights 0.01, 0.03, 0.1, 0.3 and 3 in 3D (each measured over the year and on the 2D day), all with the reference comparison if the reference grid exists), and finally `maps.sh`. Everything goes to `results/local/gapfill.csv` with the tags `sst2d_cg`, `sst2d_jac`, `sst3d_cg`, `sst3d_jac`, `acc2d` and `acc3d`.

## D10. scripts/atlas and scripts/vanda: the cluster jobs

**What is it?** `env.sh` holds the site-specific settings, `gen_jobs.sh` writes a PBS job file for each cluster experiment, and the two `.pbs` files under `vanda/` are the GPU jobs. **PBS** (Portable Batch System) is the scheduler on the NUS clusters: you hand it a script with a resource request, it places the script in a **queue** (a waiting line for a particular set of machines), and it runs the script when the requested cores are free. **Walltime** is the maximum clock time the job may take before the scheduler kills it.

**Why did we do it?** The rubric needs strong and weak scaling to at least 256 cores, which this laptop cannot supply. None of these jobs has been submitted yet: the queue names, module names and ranks-per-node below are placeholders the student must check against `gstat`, `qstat -Qf` and `module avail` on the cluster (Lecture 8).

**How does it work?** `env.sh`:

```bash
QUEUE=${QUEUE:-parallel}            # <-- replace with the CPU queue name shown by gstat
GPU_QUEUE=${GPU_QUEUE:-gpu}         # <-- replace with the GPU queue name (Vanda)
PPN=${PPN:-64}                      # MPI ranks per node; Vanda CPU nodes have 72 cores, 64 keeps powers of two
CORES_PER_NODE=${CORES_PER_NODE:-72}
WALL=${WALL:-02:00:00}
PROJECT_DIR=${PROJECT_DIR:-$HOME/poisson-mpi}
MODULES=${MODULES:-"source /app1/ebenv; module load OpenMPI"}   # Atlas example; on Vanda use the module names from 'module avail'
GPU_MODULES=${GPU_MODULES:-"module load cuda; module load openmpi"}
MPIRUN=${MPIRUN:-"mpirun"}           # some sites want 'mpirun -hostfile $PBS_NODEFILE'
```

The `${X:-default}` syntax uses an existing environment variable if set, otherwise the default, so a student can override a value for one run without editing the file. `PPN` is 64 rather than the 72 cores a Vanda node has because the rank counts in the study are powers of two, and 4 nodes times 64 is exactly 256.

`gen_jobs.sh` has a `header` function that writes the PBS preamble:

```bash
#PBS -N $name
#PBS -q $QUEUE
#PBS -l select=$nodes:ncpus=$((ppn*th)):mpiprocs=$ppn:ompthreads=$th
#PBS -l walltime=$WALL
#PBS -j oe
cd \$PBS_O_WORKDIR
$MODULES
export OMP_NUM_THREADS=$th
export OMP_PROC_BIND=close OMP_PLACES=cores
```

`-N` names the job, `-q` picks the queue, and `-l select=...` is the resource request: `select` is the number of nodes (chunks), `ncpus` the cores per node, `mpiprocs` how many MPI ranks to start per node, and `ompthreads` how many threads each rank may use. `-j oe` merges the output and error logs. `$PBS_O_WORKDIR` is the folder the job was submitted from, so the job can find `bin/`. `OMP_PROC_BIND=close` and `OMP_PLACES=cores` pin each thread to its own core so they do not wander. The script then writes `serial.pbs` (size scaling and the four serial baselines on one core), one `mpi_NP.pbs` for NP in 1 to 256 (strong and weak scaling), `design_64.pbs` and `design_256.pbs` (exchange modes and check intervals, plus the eleven process-grid shapes at 256), and one `hybrid_NPxTH.pbs` for every split of 256 cores over 4 nodes from 256x1 to 4x64. A generated hybrid job:

```bash
#PBS -N hybrid_64x4
#PBS -q parallel
#PBS -l select=4:ncpus=64:mpiprocs=16:ompthreads=4
...
MO=bin/poisson_mpi_omp
for r in 1 2; do
mpirun -np 64 --map-by ppr:16:node:pe=4 $MO --threads 4 --dim 2 --n 8192 --iters 200 --csv $OUT/hybrid.csv --tag jac2d
```

Four nodes, 16 ranks per node, 4 threads per rank, 64 ranks in total, 256 cores. `--map-by ppr:16:node:pe=4` is Open MPI's way of saying "16 processes per node, each given 4 processing elements". The cluster problem sizes are larger than the local ones (2D 8192, 3D 512 for Jacobi; 2D 4096, 3D 256 for CG) so that even at 256 ranks each rank keeps about 262 thousand points. Run with `--submit` the script calls `qsub` on each file. Without it, it only writes them so they can be read first.

`gpu.pbs` requests one node, 36 cores and one GPU, loads the CUDA and MPI modules, prints the GPU model with `nvidia-smi`, and runs `poisson_mpi_cuda` on one rank over a range of sizes, then the CPU binary `poisson_mpi_omp` with 36 threads on the same sizes, all into `gpu.csv`, so the analysis can put GPU and CPU-node times on one plot. Its comment block explains how to compile first: `qsub -I` starts an **interactive job**, a shell on a compute node where `nvcc` is available (Vanda's login nodes are virtual machines without it, Lecture 8), then `make cuda CUDA_ARCH=sm_80` with the architecture matching the GPU. `gpu_multi.pbs` requests two GPUs and runs two ranks with the `isend` and `overlap` modes and a `--cuda-aware` attempt that prints a message if the MPI library refuses device pointers, plus a weak-scaling pair.

**What came out of it?** Running `gen_jobs.sh` on the laptop with the Atlas settings produced three job files (`node_1.pbs`, `mpi_192.pbs`, `hybrid_192.pbs`), which were inspected and then deleted. None has been submitted. `results/cluster/` is empty.

**How would I explain it to a classmate in two sentences?** Edit `env.sh` with the real queue and module names, run `gen_jobs.sh --submit`, and the jobs will append their rows into `results/cluster/*.csv`. The GPU jobs have to be compiled inside an interactive job first because the login nodes have no CUDA compiler.

The data extension added ten jobs to the generator, written only if `data/india_2025.grid` is present on the cluster (the student copies it up with `scp`; the compute nodes cannot download). `data_serial.pbs` runs the four serial baselines on the data and writes the filled fields; `data_1.pbs` to `data_256.pbs` run the four data cases on each rank count, and `data_256.pbs` also runs the accuracy study. `gpu.pbs` gained four lines that run the 2D day and the 3D year on the GPU and on the CPU node. The generator was later restructured for Atlas, whose `parallel` queue allows at most 96 cores per job and 4 jobs per user: everything that fits on one 96-core node (the serial baselines, 1 to 96 ranks, the design study, the hybrid splits and the satellite runs) now goes into one job, `node_1.pbs`, and each rank count above the node size (192 on Atlas through the `hpc6a` queue; 128 and 256 on Vanda) gets its own `mpi_NP.pbs`, the largest also a `hybrid_NP.pbs`. The rank counts come from `RANKS` in `env.sh`.

## D11. analysis/summarize.cpp and analysis/plots.gp: from CSV rows to tables and figures

**What is it?** Two small files that turn the raw CSV rows into every table and figure in the report. `summarize.cpp` is a C++ program (compiled by `make` into `bin/summarize`) that reads the CSVs, keeps the best of the repeated timings, computes speedup, efficiency, communication share, convergence order and log-log slopes, writes every table as Markdown into `tables.md`, writes headline numbers into `summary.txt`, and writes plain text data files (`*.dat`) for plotting. `plots.gp` is a script for **gnuplot**, a command-line plotting program that reads columns of numbers from text files and writes image files, much as a printer takes a page description and produces paper. No Python is used anywhere in this project.

**Why did we do it?** The lecturer asked for logarithmic axes wherever a plot shows time, problem size or core count, plus speedup and efficiency numbers. Deriving these by hand from dozens of CSV rows would be slow and error-prone. Keeping the analysis in the same language as the solver means one compiler builds everything, and gnuplot is a single small program available on every cluster.

**How does it work?** The command is `make figures`. The Makefile rule is:

```make
RESULTS ?= results/local
FIGDIR  ?= results/figures
LABEL   ?= local machine

figures: $(BIN)/summarize
	mkdir -p $(FIGDIR)
	$(BIN)/summarize $(RESULTS) $(FIGDIR) "$(LABEL)"
	gnuplot -e "dir='$(FIGDIR)'" analysis/plots.gp
```

The three variables can be overridden on the command line: `make figures RESULTS=results/cluster FIGDIR=results/figures_cluster LABEL="NUS Vanda"` analyses the cluster CSVs into a separate folder. The only requirements are a C++17 compiler and gnuplot (`brew install gnuplot` on macOS, `module load gnuplot` or `apt install gnuplot` on Linux).

Inside `summarize.cpp`, each CSV row becomes a `Row`. A **struct** is a bundle of named fields kept together, like one line of a form:

```cpp
struct Row {
    std::string impl, solver, exchange, dims, hostname, tag;
    int dim = 0; long nx = 0, ny = 0, nz = 0, iters = 0; int ranks = 1, threads = 1, converged = 0, check_every = 0;
    double rel_resid = 0, l2 = 0, linf = 0, t_total = 0, t_setup = 0, tc_max = 0, th_max = 0, ta_max = 0, tc_mean = 0, th_mean = 0, ta_mean = 0, tol = 0;
    long points() const { return nx * ny * nz; }
    int cores() const { return ranks * threads; }
    std::string key() const {
        std::ostringstream k; k << impl << '|' << dim << '|' << nx << '|' << ny << '|' << nz << '|' << solver << '|' << exchange << '|' << ranks << '|' << threads << '|' << dims << '|' << check_every << '|' << tag; return k.str();
    }
};
```

`points()` and `cores()` are the two derived quantities used everywhere. `key()` joins the twelve columns that define a configuration (binary, grid, solver, exchange mode, ranks, threads, shape, check interval and tag) into one string, so two rows with the same key are repeats of the same experiment.

`split` cuts a line at commas and `load` reads one CSV file into a list of rows, skipping the header line and converting each field with `std::stoi`, `std::stol` or `std::stod` (string to integer, long integer or double). The minimum-of-repeats rule is `best`:

```cpp
// minimum t_total per configuration
static std::vector<Row> best(const std::vector<Row>& rows) {
    std::map<std::string, Row> m;
    for (const auto& r : rows) { auto it = m.find(r.key()); if (it == m.end() || r.t_total < it->second.t_total) m[r.key()] = r; }
    std::vector<Row> v; for (auto& kv : m) v.push_back(kv.second); return v;
}
```

A **map** is a lookup table from a key to a value, like a dictionary from word to definition. For each row the code looks up its key; if nothing is stored yet or the stored row was slower, this row replaces it. What remains is exactly one row per configuration, the fastest. The minimum is used rather than the mean because every disturbance on a shared machine, from another program waking up to a page being swapped, only ever adds time, so the smallest observation is the closest to the undisturbed cost.

`where` filters a list of rows by a condition. The condition is passed as a **lambda**, an unnamed function written in place, such as `[&](const Row& r) { return r.tag == tag; }`, which reads "keep the rows whose tag matches". `slope` fits the exponent of a power law:

```cpp
static double slope(const std::vector<double>& x, const std::vector<double>& y) {   // least-squares slope of log y vs log x
    if (x.size() < 2) return NAN;
    double sx = 0, sy = 0, sxx = 0, sxy = 0; size_t n = x.size();
    for (size_t i = 0; i < n; ++i) { double lx = std::log(x[i]), ly = std::log(y[i]); sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly; }
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}
```

**Least squares** means choosing the straight line that minimises the sum of the squared vertical distances to the points. Taking logarithms first turns a power law `t = c N^k` into a straight line with slope `k`, so a slope of 1 means time doubles when the grid doubles. The closed-form expression in the last line is the textbook formula for that slope.

`Out` holds the three kinds of output. Its constructor opens `tables.md` and `summary.txt` in the output folder; `dat(name)` opens a plot data file and writes a comment line naming it; `table(title, columns, rows, note)` writes a Markdown table with a `###` title. The titles are the ones the report and Part E quote, so a table can always be found by searching `tables.md` for its title.

Each `do_*` function handles one CSV. `do_validation` builds the convergence-order tables: for the `order2d` and `order3d` rows it computes the spacing `h = 1/(nx+1)`, the error ratio between successive grids, and the measured order `log(ratio) / log((n+1)/(n_prev+1))`; a value near 2 confirms second-order accuracy. It then writes the agreement table from the `agree` rows with `l2_err` printed to ten decimals so last-digit differences are visible, records the largest deviation from the serial reference in `summary.txt`, and writes the Jacobi-to-convergence table. `do_serial` writes `serial_<tag>.dat` with columns points, t_total and t_per_iter, fits the two slopes and writes the serial table with the update rate in millions of points per second. `do_strong` is the heart of the scaling analysis:

```cpp
        double t1 = ser.empty() ? par[0].t_total : ser[0].t_total;
        auto f = O.dat(std::string("strong_") + tag + ".dat"); f << "# cores t_total speedup efficiency comm_frac\n";
        for (auto& r : par) {
            double sp = t1 / r.t_total, ef = sp / r.cores(), cf = (r.th_max + r.ta_max) / r.t_total, imb = r.tc_mean > 0 ? r.tc_max / r.tc_mean : 1;
            f << r.cores() << " " << r.t_total << " " << sp << " " << ef << " " << cf << "\n";
```

`t1` is the serial binary's time for the same case (the parallel one-rank time is used only if no serial row exists). Speedup is serial time divided by parallel time, efficiency is speedup divided by cores, the communication share is the slowest rank's halo plus all-reduce time divided by total time, and `imb` is the imbalance, the slowest rank's compute time over the mean. One line per core count goes into `strong_<tag>.dat`, and the same numbers go into the strong-scaling table.

`do_weak` computes weak efficiency as the one-rank time divided by the time at p ranks (ideal 1) and writes `weak_<tag>.dat` with cores, t_total, weak_efficiency and comm_frac. `do_exchange` writes one file per strategy, `exchange_sendrecv.dat`, `exchange_isend.dat`, `exchange_overlap.dat` and `exchange_rma.dat`, each with a group index, total time, halo time and a quoted label such as `"Jacobi 2D p=16"`, and tabulates each strategy's time relative to the `isend` baseline. `do_decomp` uses `surface_cells`:

```cpp
static double surface_cells(const Row& r) {
    auto p = split(r.dims, 'x'); double px = std::stod(p[0]), py = std::stod(p[1]), pz = std::stod(p[2]);
    double lx = r.nx / px, ly = r.ny / py, lz = r.nz / pz, s = 0;
    if (px > 1) s += 2 * ly * lz; if (py > 1) s += 2 * lx * lz; if (r.dim == 3 && pz > 1) s += 2 * lx * ly; return s;
}
```

For a process-grid shape it computes the area of each rank's faces that border another rank, which is the number of values a rank sends per sweep. A 16x1 split of a 2048 grid sends 2 x 2048 values per rank; a 4x4 split sends 4 x 512 = 2048, half as many. `decomp_<tag>.dat` lists the shapes sorted by this measure with their total and halo times. `do_checkevery` writes `checkevery_<tag>.dat` with the check interval (1000 standing for "never"), total time and all-reduce time. `do_hybrid` writes `hybrid_<tag>.dat` with an index, the label such as `8x2`, total time and communication share in percent, and records the fastest split in `summary.txt`. `do_omp` writes `omp_<tag>.dat` with threads, time, speedup and efficiency relative to one thread. `do_gpu` and `do_gpu_multi` handle the cluster GPU CSVs if they exist, writing `gpu_<tag>_mpi_cuda.dat` and `gpu_<tag>_mpi_omp.dat` with points and time per iteration; they do nothing when the files are absent, as they are now.

`plots.gp` draws the figures. Its first lines set up the output:

```gnuplot
set terminal pngcairo size 1500,430 enhanced font "Helvetica,11"
set grid lc rgb "#dddddd"
set key top left box opaque
set style line 1 lc rgb "#1f77b4" pt 7 ps 1.1 lw 1.6
tags  = "jac2d jac3d cg2d cg3d"
names = "Jacobi-2D Jacobi-3D CG-2D CG-3D"
name(i) = word(names, i)
D(prefix, i) = dir."/".prefix."_".word(tags, i).".dat"
```

A gnuplot **terminal** is the output device; `pngcairo` means a PNG image of the given size. The style lines fix one colour and marker per case so the four cases look the same in every figure. `D("strong", 2)` builds the file name `results/figures/strong_jac3d.dat`, so one loop can draw all four cases. The strong-scaling block is:

```gnuplot
set output dir."/strong_scaling.png"
set multiplot layout 1,3
set logscale xy; set xlabel "cores p"; set ylabel "time (s)"; set title "Strong scaling: time"
plot for [i=1:4] D("strong",i) u 1:2 w lp ls i t name(i)
set ylabel "speedup T1/Tp"; set title "Strong scaling: speedup"
plot for [i=1:4] D("strong",i) u 1:3 w lp ls i t name(i), x ls 9 t "ideal"
unset logscale y; set yrange [0:1.3]; set ylabel "efficiency T1/(p Tp)"; set title "Strong scaling: efficiency"
plot for [i=1:4] D("strong",i) u 1:4 w lp ls i t name(i), 1 ls 9 notitle
unset multiplot
```

`multiplot layout 1,3` puts three panels side by side in one image. `set logscale xy` makes both axes logarithmic. The column selector `u 1:3` (short for `using`) says "x from column 1, y from column 3", so the speedup panel plots speedup against cores, and `w lp ls i` draws lines with points in the style of case `i`. The extra `x ls 9` term draws the straight line y = x, which is ideal speedup, in a dashed black style; in the efficiency panel the constant `1` draws the ideal efficiency line. The exchange-mode chart uses gnuplot's histogram style, in which the four strategies become four clustered bars per group:

```gnuplot
set style data histogram; set style histogram clustered gap 1; set style fill solid 0.85 border -1
set boxwidth 0.9; set logscale y; set ylabel "seconds (log scale)"; set xlabel ""
plot dir."/exchange_sendrecv.dat" u 2:xtic(4) t "sendrecv" lc rgb "#1f77b4", \
     dir."/exchange_isend.dat"    u 2 t "isend"    lc rgb "#ff7f0e", \
     dir."/exchange_overlap.dat"  u 2 t "overlap"  lc rgb "#2ca02c", \
     dir."/exchange_rma.dat"      u 2 t "rma"      lc rgb "#d62728"
```

`xtic(4)` takes the group label from column 4 of the data file, and the logarithmic vertical axis keeps the 29-second RMA bar from flattening every other bar. The GPU figure is drawn only if its data exists:

```gnuplot
if (exists_file(dir."/gpu_jac2d_mpi_cuda.dat")) {
  set output dir."/gpu_vs_cpu.png"
  ...
}
```

`exists_file` asks the shell whether the file is there, so the script runs without error before the cluster CSVs exist.

**What came out of it?** Run on the local CSVs, `make figures` writes the nine PNG files named in Part E, `tables.md` with one Markdown table per experiment, `summary.txt` with the headline numbers (convergence orders, the largest deviation from serial, the log-log slopes, speedup and efficiency at the largest core count, the best exchange mode per case and the best hybrid split), and thirty-six `.dat` files. Part E reproduces and explains each figure.

**How would I explain it to a classmate in two sentences?** A C++ program groups the CSV rows by configuration, keeps the fastest repeat, divides serial time by parallel time to get speedup, and writes tables and plain text data files. A gnuplot script then draws every figure from those files on log axes, so if a number looks wrong you re-run `make figures` and look at the matching `do_*` function.

## D12. The derived data files and summary.txt

**What is it?** The intermediate output of `bin/summarize`: thirty-six `.dat` files, one per case and experiment, and one `summary.txt`, all in `results/figures/`. A `.dat` file is plain text with one row per measurement and whitespace between columns, the simplest format gnuplot reads. An earlier version of the project wrote an Excel workbook here; it was removed because the project is C++ only, and the `.dat` files and `tables.md` carry the same derived numbers in a form anyone can open in a text editor.

**Why did we do it?** Tables in a PDF cannot be audited. A text file in which the speedup column sits next to the two times it was computed from lets a reader check any number with a calculator, and the same files are what the figures are drawn from, so a figure and its table can never disagree.

**How does it work?** Every `.dat` file starts with two comment lines, the second naming the columns. `strong_cg2d.dat` is:

```
# generated by analysis/summarize.cpp
# cores t_total speedup efficiency comm_frac
1 3.73338 1.0352 1.0352 7.74099e-05
2 1.9228 2.00998 1.00499 0.0100666
4 1.04729 3.69029 0.922574 0.0307919
8 0.566551 6.82163 0.852704 0.100213
12 0.463227 8.34321 0.695268 0.195034
16 1.03672 3.72792 0.232995 0.676167
```

To audit the 12-core row by hand, take the serial time from `summary.txt` (`strong_cg2d: ... serial_time=3.8648`) or from `results/local/strong.csv` (3.864801 s, the smaller of the two serial repeats) and divide by the 12-core time: 3.864801 / 0.463227 = 8.343, which is the third column, and 8.343 / 12 = 0.695, the fourth. The same rows appear in `tables.md` under the title "Strong scaling (strong): fixed problem, increasing cores". The other files follow the same pattern: `serial_<tag>.dat` (points, t_total, t_per_iter), `weak_<tag>.dat` (cores, t_total, weak_efficiency, comm_frac), `exchange_<mode>.dat` (group index, t_total, t_halo_max, label), `decomp_<tag>.dat` (index, dims, t_total, t_halo_max, halo_cells), `checkevery_<tag>.dat` (interval, t_total, t_allreduce_max), `hybrid_<tag>.dat` (index, label, t_total, comm_share_percent) and `omp_<tag>.dat` (threads, t_total, speedup, efficiency).

`summary.txt` is a list of `key: value` lines. The real file begins:

```
machine: Apple M4 Max laptop (12 performance + 4 efficiency cores), macOS, Open MPI 5.0.11
order_2d_last: 2.00027
...
max_l2_deviation_from_serial: 3.24379e-15
serial_slope_total_jac2d: 1.04827
serial_slope_per_iter_jac2d: 1.04827
...
strong_cg2d: max_cores=16 speedup=3.72792 efficiency=0.232995 serial_time=3.8648
...
exchange_best CG 2D p=16: overlap
hybrid_best_split_jac2d: 8x2
hybrid_best_split_jac3d: 4x4
```

The `order_` lines repeat once per grid doubling, so the last of each group is the finest-grid order. The `exchange_best` and `hybrid_best_split` lines are the automatic answers to "which strategy won", which the report text quotes.

**What came out of it?** Thirty-six plot data files and one summary, regenerated by every `make figures`. Nothing in them is typed. Everything traces to a row in a CSV.

**How would I explain it to a classmate in two sentences?** The analysis program writes plain text files where each derived number sits beside the measurements it was computed from, and gnuplot draws the figures straight from them. Divide the serial time by the parallel time yourself and you get the same speedup the report quotes.

## D13. The order in which to run everything

**What is it?** The sequence of commands that reproduces every figure and table, first on a laptop or workstation, then on the cluster.

**Why did we do it?** Reproducibility is the point of scripting. A marker or a future student should be able to regenerate the report's evidence without reading the code.

**How does it work?** Local reproduction, from inside `poisson-mpi/`:

1. `make all` (on macOS `make all CXX=clang++`) builds the four CPU binaries and `bin/gridtool`, and `make cuda_emul CXX=clang++` adds the emulation binary so the validation block can include it. A few seconds.
2. `scripts/data/download.sh india 2025 ref` downloads the year of satellite maps, the land mask and the reference (about 1.1 GB; 20 to 40 minutes depending on the connection; it can be interrupted and re-run). Then `bin/gridtool convert data/india_2025.grid --mask data/raw/india/mask.asc data/raw/india/sst_2025-*.asc` and `bin/gridtool regrid data/india_2025.grid data/india_2025_ref.grid --offset -273.15 data/raw/india/ref_2025-*.asc` build the two grid files (a minute or two). `bin/gridtool info data/india_2025.grid --days` shows what was downloaded.
3. `scripts/local/run_all.sh 16 2` runs every experiment twice on 16 cores, the data stage last. The machine should be left alone while it runs, since other work inflates the timings.
4. `make figures LABEL="Apple M4 Max, 16 cores"` compiles `bin/summarize`, runs it on `results/local/`, and runs gnuplot on `analysis/plots.gp`. It writes the figures, `tables.md`, `summary.txt` and the `.dat` files into `results/figures/`. About 5 s.
5. Build the report and handbook PDFs from their Markdown sources with `report/build.sh` and `cd tools && ./md2pdf.sh handbook_source.md ../handbook/handbook.pdf`.

Cluster reproduction:

6. Copy the folder to the cluster including the two grid files but not `data/raw/` (`scp -r poisson-mpi user@atlas8.nus.edu.sg:`), log in, `source /app1/ebenv`, load a compiler and MPI module, and `make all`.
7. Edit `scripts/atlas/env.sh`: queue name, ranks per node, module lines.
8. `scripts/atlas/gen_jobs.sh --submit` submits the CPU jobs (one single-node job plus one per multi-node rank count). `qsub scripts/vanda/gpu.pbs` and `qsub scripts/vanda/gpu_multi.pbs` submit the GPU jobs after compiling with `make cuda` inside an interactive job. Watch with `qstat`.
9. When the jobs finish, copy `results/cluster/` back to the laptop and run `make figures RESULTS=results/cluster FIGDIR=results/figures_cluster LABEL="NUS Vanda"`.
10. Replace or add the cluster figures in the report and rebuild the PDFs.

Measured durations from `results/local/run.log` on the laptop: validation 11 s; serial scaling 2 min 26 s, of which the CG 2D sweep took 70 s and CG 3D 65 s; strong scaling 35 s and weak scaling 12 s for ranks 1 to 16 (1 min 7 s together when the 12-rank point is included); exchange modes 2 min 17 s, most of it the RMA runs (the two Jacobi 3D 16-rank RMA repeats alone took 29 and 33 s); decomposition shapes 10 s; residual-check interval 9 s; hybrid splits 22 s; OpenMP threads 34 s. The whole script takes about 7 minutes on this machine, `make figures` about 5 s, and building the report PDF about 15 s.

**What came out of it?** Steps 1 to 5 have been executed on the laptop. Steps 6 to 10 (the cluster) have not been executed.

**How would I explain it to a classmate in two sentences?** Build, run the experiment script, run `make figures`, build the documents, then the same again on the cluster with the job generator in place of the local script. Each step writes into a folder the next step reads from.

## D14. Terms the brief asked about

**What is it?** The handbook brief listed ten terms to define: dataframe, solver, constraint, objective, seed, iteration, normalisation, weight, consistency ratio and Pareto front. Several come from a different kind of project (optimisation and decision analysis) and do not occur here. Each is defined below in one sentence and marked as used or not used.

**Why did we do it?** A glossary entry that pretends a term is relevant when it is not would mislead the reader. Saying plainly which terms apply keeps the handbook honest.

**How does it work?** A **dataframe** is a table held in a program's memory with named columns that can be filtered and sorted in code; the word comes from data-analysis libraries. It is not used in this project, which keeps its rows in a plain C++ list of `Row` structs (D11). A **solver** is the piece of code that takes a problem (here the grid equations) and produces its solution by some method. The two solvers are Jacobi and conjugate gradient in `solver.hpp`. An **iteration** is one repetition of a loop that improves the answer. It is used everywhere, as the `it` counter and the `iters` CSV column. **Normalisation** is dividing a quantity by a reference so it becomes a dimensionless ratio. It is used in the sense of the relative residual, which divides the residual norm by the norm of `f` so that the stopping tolerance means the same thing on every grid. A **weight** is a multiplier that says how much one input counts. It is used in the sense of the stencil coefficients `1/hx^2`, `1/hy^2`, `1/hz^2` that weight each neighbour, and nowhere else. A **constraint** is a condition a solution must satisfy in an optimisation problem. It is not used as a modelling term, though the practical limits on memory per rank and the 256-core allocation act as constraints on the experiments. An **objective** is the quantity an optimisation tries to make as small or large as possible. It is not used as a modelling term, though the practical goal of the study is the smallest wall time at a fixed accuracy. A **seed** is the starting number for a random-number generator that makes a random process repeatable. It is not used, because nothing in this project is random: the right-hand side is a formula, the initial guess is zero, and every run with the same inputs produces the same iterates. A **consistency ratio** is a measure from the Analytic Hierarchy Process of how self-contradictory a table of pairwise preference judgements is. It is not used. A **Pareto front** is the set of options in a multi-objective problem that cannot be improved in one objective without worsening another. It is not used, since the study has one objective, time, at fixed accuracy.

**What came out of it?** Four of the ten terms are used in this project and six are not. The full alphabetical glossary of terms that are used is in Part F.

**How would I explain it to a classmate in two sentences?** Solver, iteration, normalisation and weight appear in our code and mean what they usually do, while constraint and objective appear only loosely. Dataframe, seed, consistency ratio and Pareto front come from data-analysis libraries, random simulation and decision analysis and have no role in a deterministic equation solver written in C++.
# Part E. The results, explained

This Part walks through every figure and table that `make figures` (the C++ tool `bin/summarize` and the gnuplot script `analysis/plots.gp`) produces from the measurements, one section per figure or table. Each section says what the picture or table is, why we made it, how to read its axes or columns, what came out of it in real numbers, and how to tell a classmate about it in two sentences. All numbers below are copied from `results/figures/tables.md` and `results/figures/summary.txt`, which `bin/summarize` wrote from the CSV files in `results/local/`. Nothing here was typed from memory.

One honesty note before the numbers. Every timing in this Part comes from one laptop, an Apple M4 Max with 12 fast performance cores and 4 slow efficiency cores. The laptop can show how the code behaves up to 16 processes and it exposes one hardware effect very clearly, but the 256-core and GPU measurements that the grading rubric asks for have not been run yet. Section E16 says exactly which figures will change when they are.

## E1. How to read any figure in this project

**What is it?** A short guide to the conventions shared by all twelve figures: logarithmic axes, the minimum-of-repeats rule, and the difference between a "max over ranks" and a "mean over ranks" time. A logarithmic axis (log axis) is an axis on which moving the same distance always multiplies the value by the same factor, instead of adding the same amount. On a normal axis the ticks read 1, 2, 3, 4. On a log axis they read 1, 10, 100, 1000. The everyday analogy is the way we talk about money: the step from 1 dollar to 10 dollars feels about as big as the step from 1000 to 10000, and a log axis draws it that way.

**Why did we do it?** The lecturer asked for it in the project notes of lecture 12: axes that carry time, problem size or core count should be logarithmic. The reason is that our problem sizes span from 4 thousand to 17 million grid points, a factor of 4000, and the times span from 0.0004 to 31 seconds. On a normal axis the small cases would be squashed into a dot at the origin. A log axis also turns a power law into a straight line, so "time grows like size to the power 1.05" can be read off as a slope, which is exactly what the serial scaling figure does.

**How does it work?** Here is the worked example. The serial Jacobi 2D run takes 0.113 seconds on a 1024 by 1024 grid (about 1.05 million points) and 0.512 seconds on a 2048 by 2048 grid (4.19 million points). The size grew by a factor of 4.0 and the time by a factor of 4.5. On log-log axes those two points are joined by a line whose slope is log(4.5)/log(4.0) = 1.09. A slope of exactly 1 means "double the work, double the time". A slope of 2 would mean "double the work, four times the time". The script fits this slope through all the points of a series and prints it in the legend. Two other conventions matter. Every timing configuration was run twice and the script keeps the smaller of the two times (the function `best()` in `summarize.cpp` does this by keeping the first row seen for each configuration and replacing it whenever a faster repeat appears). The minimum is the right summary for a timing because everything that can disturb a measurement, such as another program waking up or the operating system moving a process, only ever makes it slower. The fastest run is the one closest to the cost of the code itself. Finally, the solver records three timers on every MPI process (compute, halo exchange, all-reduce) and reports both the largest value over all processes and the average. The wall-clock time of the program is set by the slowest process, so the "max" columns are what the user feels. The ratio of max compute time to mean compute time is printed as `imbalance` in the strong and weak scaling tables. A value of 1.00 means every process had the same amount of work and finished together, and a value of 1.46 means the slowest process needed 46 percent more time than the average one.

**What came out of it?** Nothing numerical by itself, but one reading habit that the rest of this Part relies on: when the halo or all-reduce time of a run is large while its `imbalance` is also large, the time was not spent moving data. It was spent waiting at the exchange for a slow neighbour to arrive. We will see that at 16 processes on this laptop.

**How would I explain it to a classmate in two sentences?** Our plots use log axes so that a factor of 4000 in problem size fits on one picture and so that "time proportional to size" shows up as a straight line of slope 1. We keep the fastest of two repeats, and we always report the slowest process's time because that is the time you actually wait.

## E2. Table: discretisation error, 2D and 3D

**What is it?** Two tables, produced by `do_validation()` from `validation.csv`, listing the error of the computed solution against the exact formula on grids of increasing fineness. The error is measured two ways: the discrete L2 error (the root-mean-square difference over all grid points, weighted by the cell volume) and the maximum error at any single point (Linf). The grid spacing is h = 1/(n+1), so n = 32 means h = 0.0303 and n = 1024 means h = 0.00098.

**Why did we do it?** The finite-difference method replaces a smooth curve by a polygon, and theory says the error of that replacement should shrink by a factor of four every time h is halved (this is what "second order" means). If our error did not follow that law, something in the stencil coefficients or the boundary handling would be wrong, no matter how fast the code ran. This table is the first and cheapest proof that the solver computes the right thing. It uses the serial CG solver run to a very tight tolerance (1e-10), so that the error we see is the discretisation error and not leftover iteration error.

**How does it work?** Each row is one grid. The `ratio` column is the L2 error of the previous row divided by the L2 error of this row, and `order` is the exponent that turns the change in h into that ratio. Read down the column: a ratio of 4.00 and an order of 2.00 are the textbook values. The `iters` column shows how many CG iterations the tight tolerance needed, and `rel_resid` confirms that the linear system itself was solved to about one part in 10 billion.

**What came out of it?** In 2D the L2 error goes 5.266e-4, 1.357e-4, 3.445e-5, 8.680e-6, 2.178e-6, 5.457e-7 for n = 32, 64, 128, 256, 512, 1024, giving ratios 3.88, 3.94, 3.97, 3.98, 3.99 and a measured order of 2.00 at every step (`summary.txt` records 2 for the finest pair, to the six figures it prints). In 3D the errors go 5.406e-3, 1.511e-3, 4.005e-4, 1.032e-4, 2.620e-5 for n = 8, 16, 32, 64, 128, with ratios 3.58, 3.77, 3.88, 3.94. The 3.58 on the coarsest 3D grid is not a defect: with only 8 points per side (h = 0.111) the grid is too coarse for the leading error term to dominate, so higher-order terms still contribute, and the ratio creeps up towards 4 as the grid is refined, exactly as theory predicts. The maximum errors follow the same pattern (1.110e-3 down to 1.153e-6 in 2D). The CG iteration counts, 93, 187, 371, 738, 1467 in 2D, double when n doubles, which is the O(n) behaviour expected of conjugate gradient on this matrix.

**How would I explain it to a classmate in two sentences?** Every time we make the grid twice as fine, the error against the exact answer drops by almost exactly four, in both 2D and 3D, which is what second-order finite differences are supposed to do. That is our evidence that the equations in the code are right before we talk about speed at all.

## E3. Table: agreement of every implementation

**What is it?** A table, also from `do_validation()`, in which the same four problems (Jacobi 2D on a 300 by 300 grid for exactly 300 sweeps, Jacobi 3D on 45 cubed for 300 sweeps, CG 2D on 300 by 300 to tolerance 1e-9, CG 3D on 45 cubed to 1e-9) were run through every implementation we have: the serial binary, the OpenMP binary with 4 threads, pure MPI with each of the four exchange modes on 6 or 12 processes, MPI+OpenMP with 2 threads per process, and the CUDA driver through its CPU emulation layer on 4 to 8 processes. The columns show the iteration count, the final relative residual, and the L2 and maximum errors to ten significant digits.

**Why did we do it?** A parallel program can be fast and wrong. The classic bugs in domain decomposition are off-by-one errors at block boundaries, halos that are refreshed one sweep too late, and faces packed in the wrong order. Each of those would change the answer slightly, often by an amount too small to notice by eye. Running identical problems through every code path and comparing to many digits catches all of them at once. It also tests the decomposition on awkward process counts (6, 7, 12) where the blocks do not divide evenly and some ranks own one more row than others.

**How does it work?** Group the rows by the `case` column and compare the error columns within a group. For the fixed-sweep Jacobi cases the arithmetic performed on every grid point is identical regardless of how the grid is cut, so the printed digits should be identical. Only the final diagnostic sums are added in a different order. For the CG cases the dot products are summed in a different order on each process count, which changes the rounding in the last digits of alpha and beta, so the residual and the iteration path may differ at the level of 1e-13 while the error against the exact solution agrees to about ten digits.

**What came out of it?** Every row agrees with the serial reference. The Jacobi 2D case gives L2 error 8.3708880689e-01 and maximum error 1.7034276923e+00 for all nine implementations, including the emulated CUDA path in both normal and overlap mode. The Jacobi 3D case gives 2.9523202694e-01 for all seven. For CG 2D every implementation stops at 432 iterations with L2 error 6.3277141857e-06 to 6.3277141917e-06, and for CG 3D at 76 iterations with 2.0610511554e-04. `summary.txt` records the largest deviation of any implementation's L2 error from the serial value for each case: 3.2e-15 for CG 2D, 1.7e-15 for CG 3D, 1.2e-15 for Jacobi 2D and 1.9e-15 for Jacobi 3D. Those are a few units in the sixteenth decimal place, the size of a single rounding error in double precision. There is no discrepancy to explain.

**How would I explain it to a classmate in two sentences?** We ran the exact same problem through the serial code, the OpenMP code, MPI in four flavours, MPI plus OpenMP and the emulated GPU code, and the answers agree to fifteen decimal places. The parallel versions are not approximately right, they are the same computation.

## E4. Table: Jacobi to convergence, serial vs MPI

**What is it?** A four-row table from `do_validation()` in which Jacobi was run not for a fixed number of sweeps but until the relative residual fell below 1e-8, checking every 50 sweeps, once in serial and once on 4 processes (2D, n = 64) or 8 processes (3D, n = 24).

**Why did we do it?** The agreement table in E3 uses a fixed sweep count, which tests the sweep itself. This table tests the stopping logic, which involves the one piece of global communication Jacobi has: the residual norm, summed over all processes with an all-reduce and compared to the tolerance on every rank. If that sum were wrong on any rank, or if a rank used stale halos when computing its residual, the parallel run would stop at a different sweep than the serial one.

**How does it work?** Compare the `iters` and `rel_resid` columns between the serial and MPI rows of each dimension. They should match exactly, because the convergence test is evaluated on the same global numbers.

**What came out of it?** In 2D both runs stop at sweep 15700 with relative residual 9.647e-9 and L2 error 1.3569e-4. In 3D both stop at sweep 2350 with residual 7.766e-9 and L2 error 6.9806e-4. The L2 error of 1.3569e-4 at n = 64 is also the value CG produced in E2 for the same grid (1.357e-4), which confirms that both solvers converge to the same discrete solution. As a side note, 15700 sweeps for a 64 by 64 grid is why Jacobi is used as a benchmark here and not as the production solver: CG needed 93.

**How would I explain it to a classmate in two sentences?** When Jacobi is told to stop at a tolerance, the serial code and the 4- or 8-process code stop at exactly the same sweep with the same residual. The global convergence check works in parallel.

## E5. Figure serial_scaling.png and its table

![`results/figures/serial_scaling.png`, produced by `analysis/summarize.cpp` and `analysis/plots.gp`.](../results/figures/serial_scaling.png){width=100%}

**What is it?** Two panels produced by `do_serial()` from `serial_scaling.csv`, plus a table. The left panel plots the total time of the serial binary against the number of grid points N for the four cases (Jacobi 2D and 3D for 200 fixed sweeps, CG 2D and 3D to tolerance 1e-8). The right panel plots time per iteration. Both panels use log-log axes and print the fitted slope in the legend. The table adds the time per iteration in milliseconds and a throughput column, `Mupdates_per_s`, the number of grid-point updates performed per second in millions.

**Why did we do it?** The grading rubric awards marks for a serial runtime scaling analysis against problem size. Beyond the marks, this figure is the baseline for everything else: it tells us the cost of one sweep as a function of N, and whether that cost is limited by arithmetic or by moving data through memory. It also exposes the difference between Jacobi and CG: one has a fixed sweep count, the other needs more iterations as the grid grows.

**How does it work?** On the right panel, a slope of 1 means the time per iteration is proportional to N, as it must be when every point costs the same. On the left panel the slope adds the growth of the iteration count. For CG the iteration count grows like n, the number of points per side, which is N to the power 1/2 in 2D and N to the power 1/3 in 3D, so the total time should grow like N to the power 1.5 in 2D and 1.33 in 3D. In the table, `Mupdates_per_s` is N times iterations divided by time. For a memory-bound kernel it should be roughly constant across sizes and it lets us estimate the memory bandwidth the single core achieves.

**What came out of it?** The per-iteration slopes are 1.05 (Jacobi 2D), 1.01 (Jacobi 3D), 1.01 (CG 2D) and 1.00 (CG 3D): linear in N, as expected. The total-time slopes for Jacobi are the same numbers because its sweep count is fixed, and for CG they are 1.51 in 2D and 1.35 in 3D, within a few hundredths of the theoretical 1.5 and 1.33. The throughput of Jacobi is 1640 to 2300 million updates per second (2300 on the smallest grids that fit in cache, about 1700 on the 16.8-million-point grids), and CG runs at a steady 396 to 430 million point-iterations per second. These rates tell us the kernels are limited by memory traffic, not arithmetic. A Jacobi update reads the source value f, reads one new value of the old array (the four or six neighbours are mostly reused from cache), and writes one value of the new array: about 24 bytes per update. At 2000 million updates per second that is about 48 gigabytes per second streamed by one core, a large fraction of what one core can pull from this laptop's memory. One CG iteration touches each point roughly six times across its vector operations (the stencil application, two dot products, two axpy updates and the direction update), on the order of 100 bytes per point, so 400 million point-iterations per second is again about 40 gigabytes per second. The largest single run in this table is CG 2D on 2048 by 2048: 2909 iterations in 30.85 seconds.

**How would I explain it to a classmate in two sentences?** On one core, the time per sweep grows exactly in proportion to the number of grid points, and CG's total time grows a bit faster because it also needs more iterations on finer grids. Both solvers are limited by how fast the core can stream numbers from memory, about 40 to 50 gigabytes per second here, not by how fast it can multiply.

## E6. Figure strong_scaling.png and its table

![`results/figures/strong_scaling.png`, produced by `analysis/summarize.cpp` and `analysis/plots.gp`.](../results/figures/strong_scaling.png){width=100%}

**What is it?** Three panels from `do_strong()` using `strong.csv`: time, speedup and parallel efficiency against the number of cores p for a fixed problem (Jacobi 2D on 2048 by 2048 and Jacobi 3D on 192 cubed, both for 200 sweeps, and CG 2D on 1024 by 1024 and CG 3D on 128 cubed, both to tolerance 1e-8). Speedup is the serial time divided by the parallel time. Efficiency is speedup divided by p. The table adds the process grid `dims`, the compute, halo and all-reduce times of the slowest process, the `imbalance` ratio and `comm_frac`, the share of the run spent in communication. Rank counts are 1, 2, 4, 8, 12 and 16. The 12 was added so that we could see the laptop's 12 performance cores on their own.

**Why did we do it?** Strong scaling is the question everybody asks of a parallel code: if I have a fixed job and buy more cores, how much faster does it finish? It is the headline measurement of the rubric. Running the MPI binary on one process alongside the serial binary also measures the overhead of MPI itself.

**How does it work?** In the middle panel the dashed line is ideal speedup, equal to p. A curve on it means perfect scaling. A curve that bends below it means the extra cores are partly wasted. A curve that turns downward means adding cores made the job slower. The right panel shows the same information as efficiency, with 1.0 as the ideal. In the table, read `t_compute_max`, `imbalance` and `comm_frac` together: when compute time stops falling as p grows, the cores are competing for memory bandwidth. When `imbalance` jumps above 1.1, some processes are slower than others and the halo and all-reduce timers mostly record waiting.

**What came out of it?** On one process the MPI binary matches the serial binary within 3 to 4 percent (the table's speedup of 1.03 or 1.04 at p = 1 says the MPI build was in fact marginally faster on this run, which is noise). Up to 8 cores all four cases scale usefully. CG 2D reaches a speedup of 6.82 at 8 cores (85 percent efficiency) and 8.34 at 12 cores (70 percent). CG 3D reaches 5.95 at 8 (74 percent) and 6.54 at 12 (55 percent). Jacobi behaves differently: Jacobi 2D reaches 4.02 at 4 cores (100 percent), then only 4.88 at 8 and 4.83 at 12. Look at its `t_compute_max` column: 0.125 s at 4 cores, 0.102 at 8, 0.098 at 12. Adding cores beyond 4 barely reduces the compute time, even though each core now holds half or a third as many points. That is memory bandwidth saturation: the Jacobi sweep streams about 24 bytes per update, and eight cores together already pull as much from the memory system as it can deliver, so more cores cannot stream faster. CG, which does about four times more arithmetic per byte in its dot products and vector updates, saturates later. Jacobi 3D shows the same plateau (compute 0.124 s at 8, 0.112 at 12) with a larger halo share because its blocks have six faces.

At 16 cores every case gets slower, not faster: Jacobi 2D drops from 4.83x to 2.60x, CG 2D from 8.34x to 3.73x, Jacobi 3D from 3.38x to 2.25x, CG 3D from 6.54x to 2.88x. The `imbalance` column explains it: 1.03 or less up to 12 cores, then 1.46, 1.28, 1.14 and 1.15 at 16. Four of the sixteen processes landed on the laptop's efficiency cores, which run the same work in about one and a half times the time. Everyone else finishes early and then waits for them at the next halo exchange or all-reduce, so the halo time of Jacobi 2D jumps from 0.016 s at 12 cores to 0.150 s at 16 while the compute time of the slowest process stays at 0.100 s, and the all-reduce time of CG 2D jumps from 0.070 s to 0.610 s. Those are waiting times, not transfer times. The 16-core points are therefore a measurement of this laptop, not of the algorithm, and they are the strongest argument for running the same scripts on a cluster where all cores are alike.

**How would I explain it to a classmate in two sentences?** On the 12 identical fast cores, conjugate gradient in 2D gets 8.3 times faster and Jacobi flattens at about 5 times because it runs out of memory bandwidth, not cores. At 16 processes four of them sit on slow efficiency cores and everyone waits for them, so the whole run slows down. The imbalance column in the table is how we know.

## E7. Figure strong_comm_fraction.png

![`results/figures/strong_comm_fraction.png`, produced by `analysis/summarize.cpp` and `analysis/plots.gp`.](../results/figures/strong_comm_fraction.png){width=100%}

**What is it?** A single panel from `do_strong()` showing, for the same four strong-scaling cases, the percentage of the slowest process's time spent in the halo exchange plus the all-reduce, against the number of cores on a log axis.

**Why did we do it?** The rubric asks us to explain the impact of communication on the runtime. The speedup curves show that scaling degrades. This figure shows how much of the degradation is attributable to the communication timers and how that share grows with the process count. It is also the figure that will change most on a cluster, where the network replaces shared memory.

**How does it work?** Each point is `comm_frac` from the strong-scaling table: (halo max plus all-reduce max) divided by total time. A rising curve is expected for strong scaling, because each process's block shrinks as p grows while the number of messages per sweep stays the same, so the fixed cost of a message becomes a larger share. What a rising curve cannot tell you on its own is whether the time went into moving bytes or into waiting. For that, read it next to the `imbalance` column.

**What came out of it?** The share starts near zero on one process (there is no communication), sits between 1 and 15 percent at 4 cores, and between 10 and 41 percent at 8 cores (Jacobi 3D highest at 41 percent, because its 200 sweeps on a 192 cubed grid are short, 0.19 s in total, while it exchanges six faces per sweep). At 12 cores the shares are 15.6 percent (Jacobi 2D), 19.5 percent (CG 2D), 29.9 percent (CG 3D) and 47.1 percent (Jacobi 3D). Then they jump to 67 to 82 percent at 16 cores. The jump is not communication in the ordinary sense. From E6 we know the 16-core runs have imbalance 1.14 to 1.46, so most of the "communication" time is processes on fast cores waiting at `MPI_Waitall` or `MPI_Allreduce` for processes on slow cores. The genuine communication trend is the smooth rise up to 12 cores.

**How would I explain it to a classmate in two sentences?** Communication grows from about one percent of the time on 2 cores to 15 to 47 percent on 12 cores, because each process's block shrinks while it still sends the same number of messages per sweep. The jump to 70 or 80 percent at 16 is mostly fast cores waiting for the four slow ones, which the imbalance numbers prove.

## E8. Figure weak_scaling.png and its table

![`results/figures/weak_scaling.png`, produced by `analysis/summarize.cpp` and `analysis/plots.gp`.](../results/figures/weak_scaling.png){width=100%}

**What is it?** Two panels from `do_weak()` using `weak.csv`: total time and weak efficiency against cores, where the grid grows with the process count so that each process always owns the same number of points (1024 by 1024 points per process for Jacobi 2D, 64 cubed for Jacobi 3D, 512 by 512 for CG 2D and 48 cubed for CG 3D). Jacobi runs 200 sweeps and CG runs a fixed 300 iterations, because running CG to a tolerance would need more iterations on the larger global grids and would mix two effects. Weak efficiency is the one-process time divided by the p-process time. The ideal is a flat line at 1.

**Why did we do it?** Strong scaling eventually fails for any code because the work per process goes to zero. Weak scaling asks the other natural question: if I have a bigger problem and proportionally more cores, does it take the same time? This is how supercomputers are actually used, and it is the other half of the rubric's scaling requirement. The lecturer also pointed out that synthetic data makes weak scaling easy, because we can generate any grid size we like with the `--local-n` option.

**How does it work?** A flat time curve means perfect weak scaling. Any rise is either the growing cost of communication (more neighbours, longer all-reduce chains) or, on a single machine, the shrinking memory bandwidth available to each process as more of them compete. The `global_grid` column shows how the grid grew: for example CG 2D goes 512x512, 1024x512, 1024x1024, 2048x1024, 2048x1536, 2048x2048 for 1, 2, 4, 8, 12, 16 processes.

**What came out of it?** Up to 4 cores every case stays above 0.73 efficiency and the CG cases above 0.90. At 8 cores the efficiencies are 0.49 (Jacobi 2D), 0.58 (Jacobi 3D), 0.76 (CG 2D) and 0.83 (CG 3D). At 12 cores they are 0.34, 0.40, 0.49 and 0.67, and at 16 cores they fall to 0.12, 0.11, 0.14 and 0.20 with imbalance 1.54, 1.24, 1.32 and 1.09, the same efficiency-core effect as in E6. The interesting comparison is Jacobi against CG at 8 and 12 cores. Jacobi is worse even though it communicates less (it has no all-reduce). The reason is the one found in E5 and E6: Jacobi is the more bandwidth-hungry kernel per unit of work, and in a weak-scaling run on one machine the total amount of data streamed grows in proportion to p while the memory system does not. Each process gets a smaller share of the bandwidth, so its fixed amount of work takes longer. CG does more arithmetic per byte and feels this later. On a cluster the picture would be different: there each node brings its own memory system, so the bandwidth per process stays constant as nodes are added, and what grows instead is the network cost of the halo exchange and the all-reduce. The cluster weak-scaling runs are therefore not a repeat of this figure but a different experiment, and they are the ones that answer the rubric's question.

**How would I explain it to a classmate in two sentences?** With the same work per process, CG stays within 70 to 80 percent of ideal up to 8 cores while Jacobi falls to 50 percent, because Jacobi needs more memory bandwidth per unit of work and 8 processes on one laptop have to share it. On a cluster every node brings its own memory, so the limiting factor there will be the network instead, which is what the cluster scripts measure.

## E9. Figure exchange_modes.png and its table

![`results/figures/exchange_modes.png`, produced by `analysis/summarize.cpp` and `analysis/plots.gp`.](../results/figures/exchange_modes.png){width=100%}

**What is it?** Two bar charts from `do_exchange()` using `exchange.csv`: total time and halo-exchange time for the four halo exchange strategies (`sendrecv`, `isend`, `overlap`, `rma`) on 8 and 16 processes, for Jacobi 2D on 2048 by 2048, Jacobi 3D on 192 cubed and CG 2D on 1024 by 1024 with a fixed 300 iterations. The y axis is logarithmic because one bar is a hundred times taller than the others. The table lists the compute, halo and all-reduce times and `vs_isend`, the time relative to the non-blocking baseline.

**Why did we do it?** How halos are exchanged is the central design decision of a domain-decomposition code, and the lecture 12 slides pose it as an open question (sends and receives, or windows). We implemented four answers so that the choice is a measurement rather than a guess. `isend` is the baseline because posting all receives, then all sends, then waiting once is the textbook non-blocking pattern.

**How does it work?** Compare bars within one group (one case at one process count). A `vs_isend` below 1.00 means the strategy beat the baseline. The `overlap` strategy can only win if the deep-interior computation takes longer than the messages, and it pays a price for computing the boundary layer in six separate passes afterwards. `rma` uses one-sided `MPI_Put` inside a pair of `MPI_Win_fence` calls. Its cost depends heavily on how the MPI library implements fences.

**What came out of it?** At 8 processes `sendrecv`, `isend` and `overlap` are within 11 percent of each other in every case: for CG 2D the times are 0.123, 0.132 and 0.125 s, for Jacobi 2D 0.119, 0.114 and 0.106 s, for Jacobi 3D 0.157, 0.176 and 0.214 s. In 2D, `overlap` is the best or close to it (0.94 and 0.90 times the baseline for Jacobi at 8 and 16 processes, and 0.63 for CG at 16 processes, where hiding the exchange behind the interior work also absorbed some of the waiting for slow ranks). In 3D Jacobi `overlap` is worse, 1.22 times the baseline at 8 processes and 1.16 at 16: its compute time rises from 0.119 to 0.150 s because the six boundary slabs of a 3D block are small, strided loops that run far below the streaming speed of the single full sweep, and that loss is larger than the 0.06 s of halo time it was trying to hide. `sendrecv` is as good as `isend` at 8 processes (0.89 to 1.04) but worse at 16 (1.35 to 1.95), where serialising the dimensions means each blocking call waits for the slowest neighbour in turn.

`rma` is the clear loser on this machine: 3.1 to 7.3 times slower than the baseline at 8 processes (halo time 0.26 s for CG 2D, 0.48 s for Jacobi 2D, 1.15 s for Jacobi 3D, against 0.004 to 0.063 s for `isend`), and the Jacobi 3D run at 16 processes took 28.98 s, of which 28.99 s was in the halo timer, against 0.354 s for `isend`. We state this plainly: one-sided communication with `MPI_Win_fence` in Open MPI 5.0.11 over macOS shared memory performs badly here, and the 29-second run looks like a progress or synchronisation pathology in that implementation rather than a property of the algorithm. On a cluster with an InfiniBand network and hardware support for remote memory access, one-sided transfers are designed to be efficient, and the result could reverse. This is one of the questions the cluster run has to settle. Until then, `rma` should not be used.

**How would I explain it to a classmate in two sentences?** Non-blocking sends and receives are the safe default. Overlapping computation with the messages helps a little in 2D (up to 37 percent at 16 processes) but hurts in 3D because the leftover boundary loops are slow. One-sided MPI with fences was 3 to 80 times slower on this laptop's MPI library, so we would only revisit it on a real cluster network.

## E10. Figure decomposition_shape.png and its table

![`results/figures/decomposition_shape.png`, produced by `analysis/summarize.cpp` and `analysis/plots.gp`.](../results/figures/decomposition_shape.png){width=100%}

**What is it?** Two bar charts from `do_decomp()` using `decomp.csv`, for Jacobi on 16 processes: total time and halo time for several shapes of the process grid, forced with the `--dims` option. In 2D (2048 by 2048 grid) the shapes are 4x4, 2x8, 8x2, 16x1 and 1x16. In 3D (192 cubed) they are 1x4x4, 2x2x4, 4x2x2, 4x4x1, 16x1x1 and 1x1x16. The table adds `halo_cells_per_rank`, the number of boundary values each process must send per sweep, computed by the `surface_cells()` function from the block dimensions.

**Why did we do it?** The textbook argument says a process grid should make the blocks as cube-like as possible, because a cube has the least surface for its volume and the surface is what gets sent. Our data layout adds a second consideration that the textbook argument ignores: the x direction is contiguous in memory, so a face perpendicular to x is a set of scattered values (one every row), while a face perpendicular to z is one solid run of memory. We wanted to know which effect matters more at this scale.

**How does it work?** Within each panel, compare bar heights. The dark part of each bar is the halo time. In the table, group the rows by `halo_cells_per_rank`: shapes with the same number have the same communication volume, so any difference between them must come from something other than volume.

**What came out of it?** In 2D all five shapes lie within 19 percent of each other (0.190 to 0.226 s). The two best are 4x4 (2048 halo cells, 0.190 s) and 1x16 (4096 halo cells, 0.191 s): the square blocks send half as much, the 1x16 slabs send twice as much but each face is one contiguous row. The worst is 8x2 at 0.226 s. In 3D the volume argument alone fails outright. The two slab shapes both send 73728 cells per process, yet 16x1x1 is the slowest at 0.420 s (halo 0.351 s) and 1x1x16 is the fastest at 0.295 s (halo 0.216 s). The difference is entirely layout: slabs cut along x have faces made of 192 times 192 scattered values at stride 194, which MPI must gather element by element through the subarray datatype, while slabs cut along z have faces that are one contiguous block of 36864 doubles. The four block-like shapes, each sending 36864 cells (half the volume), sit in between at 0.284 to 0.334 s. At this problem size and process count, contiguity of the faces matters more than their total size. A good default is therefore to let `MPI_Dims_create` choose a compact grid but, if one direction must be left unsplit, to leave the contiguous direction unsplit.

**How would I explain it to a classmate in two sentences?** Cutting the grid into compact blocks does send the least data, but on 16 processes the fastest 3D layout was the one whose faces were contiguous in memory, even though it sent twice as much. Message count and memory layout beat raw byte count at this scale.

## E11. Figure check_interval.png and its table

![`results/figures/check_interval.png`, produced by `analysis/summarize.cpp` and `analysis/plots.gp`.](../results/figures/check_interval.png){width=100%}

**What is it?** A line chart from `do_checkevery()` using `checkevery.csv`: total time and all-reduce time for Jacobi on 16 processes (2D on 2048 by 2048, 3D on 192 cubed, 200 sweeps) as the residual check interval `--check-every` is set to 1, 10, 100 and 0 (never, plotted at the right-hand end).

**Why did we do it?** Jacobi needs to compute the global residual to know when to stop, and that is its only global synchronisation. Each check costs an extra halo exchange (the residual needs fresh halos of the new iterate), one extra pass over the block, and one `MPI_Allreduce`. We wanted to measure how expensive checking is, so that the stopping logic can be designed with numbers rather than instinct.

**How does it work?** Read the solid lines for total time and the dotted lines for all-reduce time. Checking every sweep is the left end. Never checking is the right end. The difference between them is the entire cost of the convergence test.

**What came out of it?** Checking every sweep costs 2.6 times the total time in 2D (0.566 s against 0.216 s with no checks) and 2.2 times in 3D (0.695 against 0.313 s). The all-reduce time alone rises from 0.012 to 0.345 s in 2D and from 0.017 to 0.320 s in 3D, and the compute time rises from 0.087 to 0.230 s because the residual pass is as expensive as a sweep. Checking every 10 sweeps already removes almost all of the cost (0.205 s in 2D, within noise of never checking, and 0.344 s in 3D, 10 percent above never), and every 100 sweeps is indistinguishable from never. The recommendation is to check every 10 to 100 sweeps. The code uses 10 as its default when a tolerance is given and 0 for the fixed-sweep benchmarks.

**How would I explain it to a classmate in two sentences?** Testing for convergence after every Jacobi sweep more than doubles the run time on 16 processes, because each test is a global sum plus an extra pass over the data. Testing every 10 sweeps costs almost nothing, so that is what we do.

## E12. Figure hybrid_split.png and its table

![`results/figures/hybrid_split.png`, produced by `analysis/summarize.cpp` and `analysis/plots.gp`.](../results/figures/hybrid_split.png){width=100%}

**What is it?** Two panels from `do_hybrid()` using `hybrid.csv`, for the MPI+OpenMP binary with the total core count fixed at 16 and divided five ways: 16 processes with 1 thread each, 8x2, 4x4, 2x8 and 1x16 (one process, pure OpenMP). The left panel shows total time. The right shows the communication share. All four cases are included (Jacobi 2D on 2048 by 2048, Jacobi 3D on 192 cubed, CG 2D on 1024 by 1024 to 1e-8, CG 3D on 128 cubed to 1e-8).

**Why did we do it?** This is the question the lecture 9 slides ask directly: on a 72-core Vanda node, is it better to run one MPI process with 72 threads, 72 processes with one thread, or something between? Processes communicate through MPI messages and each owns a separate block. Threads share the block and communicate through memory. Fewer processes means fewer and larger messages, but more threads means more fork-join overhead inside every loop. Only a measurement can say where the balance lies for this code on this machine.

**How does it work?** Read along the x axis from all-processes to all-threads and find the minimum of each curve. The right panel shows why the curves move: as the process count falls, the communication share falls with it.

**What came out of it?** 16x1 is the worst split in every case (0.214, 0.306, 1.186 and 0.431 s), partly because of the efficiency-core waiting discussed in E6, and its communication share is 69 to 76 percent. Moving to 8x2 cuts the time sharply: 0.114 s for Jacobi 2D (1.9 times faster than 16x1), 0.180 s for Jacobi 3D, 0.780 s for CG 2D and 0.217 s for CG 3D, with communication shares down to 21 to 49 percent. For Jacobi 2D, CG 2D and CG 3D the 8x2 split is the fastest of the five. For Jacobi 3D the picture is flat: 4x4 and 1x16 both take 0.170 s and 8x2 takes 0.180 s, a 6 percent spread that is within the noise of this measurement, so for that case we can only say that anything from 8 processes down is fine. Pure OpenMP (1x16) is never the best for the CG cases (1.022 and 0.274 s against 0.780 and 0.217 s for 8x2): a single process with 16 threads pays the fork-join cost of a parallel region on every one of the six vector operations in each CG iteration, roughly 9000 regions for the 1458 iterations of the 2D case, and it also leaves only one process to drive memory traffic. The lesson for a 72-core node is that neither extreme is right: a handful of processes per node (one per memory controller or per 8 to 16 cores) with threads filling the rest is the place to start, and the exact best split must be measured there, which `scripts/atlas/gen_jobs.sh` does with its seven hybrid jobs (256x1 through 4x64 over four nodes).

**How would I explain it to a classmate in two sentences?** With 16 cores, 8 MPI processes each running 2 threads beat both 16 single-threaded processes (by up to 1.9 times) and one process with 16 threads, in three of the four cases, with the fourth a tie. Fewer processes mean fewer messages, but one process with many threads pays for a thread fork on every loop, so the sweet spot is in the middle.

## E13. Figure openmp_threads.png and its table

![`results/figures/openmp_threads.png`, produced by `analysis/summarize.cpp` and `analysis/plots.gp`.](../results/figures/openmp_threads.png){width=100%}

**What is it?** Two panels from `do_omp()` using `omp_threads.csv`: speedup against thread count for the OpenMP-only binary (no MPI at all) with 1, 2, 4, 8 and 16 threads on the same four problems as E6, with the MPI strong-scaling speedups overlaid as dashed lines for comparison, and the OpenMP parallel efficiency on the right. Speedup here is relative to the OpenMP binary running one thread, not to the serial binary.

**Why did we do it?** This isolates the shared-memory half of MPI+OpenMP. If OpenMP threads alone scaled perfectly inside a node, the hybrid question would be trivial (use one process per node). The comparison with MPI on the same cores shows which model copes better with this laptop's memory system and heterogeneous cores.

**How does it work?** Left panel, log-log: the dashed black line is ideal. Solid lines are OpenMP. Faint dashed lines are the MPI runs from E6. Right panel: efficiency against threads. Note the baseline difference: the one-thread OpenMP binary is slower than the serial binary (0.645 against 0.515 s for Jacobi 2D), because the compiled OpenMP loops carry the parallel-region machinery even for one thread, so OpenMP speedups measured against the serial binary would be about 20 percent lower than the numbers in this table.

**What came out of it?** OpenMP scales almost perfectly to 2 threads (efficiency 0.96 to 0.98 in every case) and well to 4 (0.82 to 0.87), then falls to 0.55 to 0.63 at 8 threads and 0.27 to 0.33 at 16. The speedups at 16 threads are 5.08 (Jacobi 2D), 5.29 (Jacobi 3D), 4.31 (CG 2D) and 4.84 (CG 3D). CG 2D actually gets slower from 8 to 16 threads (0.976 to 1.022 s). Two mechanisms limit it, the same two seen with MPI: memory bandwidth, which caps Jacobi at about 5 times no matter how the cores are used, and the four efficiency cores, which with static scheduling receive the same share of each loop as the fast cores and finish last, so every parallel region ends with twelve threads idling. Compared with MPI on the same number of cores, OpenMP is slightly behind up to 8 (at 8, Jacobi 2D 4.48 against 4.88 for MPI, and CG 2D 4.52 against 6.82) because MPI processes each stream their own memory while threads share one process's address space and pay a fork-join at every loop. At 16, OpenMP is ahead (5.08 against 2.60 for Jacobi 2D) because a slow thread delays one loop while a slow MPI process delays every message exchange it takes part in. Measured against the serial binary, the 16-thread Jacobi 2D run (0.127 s) is 4.1 times faster than serial (0.515 s).

**How would I explain it to a classmate in two sentences?** Threads alone give about 5 times on 16 cores for Jacobi and 4.3 to 4.8 times for CG, limited by memory bandwidth and by the four slow cores that make the other twelve wait at the end of every loop. MPI processes are a bit faster up to 8 cores but suffer more from the slow cores at 16, which is why mixing the two won in E12.

## E14. The derived data files

**What is it?** The plain text files in `results/figures/` from which every figure in this Part is drawn: one `.dat` file per case and experiment, plus `summary.txt`. Part D12 describes their layout.

**Why did we do it?** Tables in a PDF cannot be audited. A text file in which the speedup sits next to the times it came from lets any reader check a number with a calculator, and because the figures are drawn from the same files, a figure and its table cannot disagree.

**How does it work?** Take the 12-core row of `strong_cg2d.dat`: `12 0.463227 8.34321 0.695268 0.195034`. The serial time for the same case is 3.864801 s (the smaller of the two serial repeats in `results/local/strong.csv`, also shown rounded in `summary.txt` as `serial_time=3.8648`). Then 3.864801 / 0.463227 = 8.343, the speedup in column 3, and 8.343 / 12 = 0.695, the efficiency in column 4, both matching the table in E6 (8.34 and 0.70) to rounding. The last column, 0.195, is the 19.5 percent communication share in the same table.

**What came out of it?** The same numbers as the tables in this Part, in a form that can be checked line by line. Nothing in these files is typed. Everything traces to a row in a CSV.

**How would I explain it to a classmate in two sentences?** The analysis writes plain text files with the measurements and the derived numbers side by side, and the figures are drawn from them. Divide the serial time by the parallel time yourself and you get the speedup the report quotes.

## E15. The final recommendation and the honest caveats

**What is it?** The configuration we would carry to the cluster, and the list of things these local results cannot tell us.

**Why did we do it?** A measurement campaign should end in a decision. The rubric's discussion marks also ask us to say how we chose to parallelise, which parts needed the most communication and what we learned.

**How does it work?** Each recommendation below points to the section whose numbers justify it.

**What came out of it?** For the exchange strategy, use `isend` as the default. In 2D, `overlap` is worth enabling (E9: 0.90 to 0.94 of the baseline at 8 to 16 processes and 0.63 for CG at 16), in 3D stay with `isend` or `sendrecv` (overlap cost 16 to 22 percent). Do not use `rma` until it has been measured on the cluster network (E9). Let `MPI_Dims_create` pick a compact process grid and never split only the contiguous x direction (E10: 16x1x1 was 42 percent slower than 1x1x16 for the same byte count). Check Jacobi's residual every 10 to 100 sweeps, never every sweep (E11: 2.2 to 2.6 times slower). Within a node, do not run one MPI process per core and do not run one process per node either. Start at 8 to 16 processes per node with threads filling the remaining cores and measure the splits (E12: 8x2 beat 16x1 by up to 1.9 times and beat 1x16 in the CG cases). Expect Jacobi to stop scaling once the node's memory bandwidth is saturated (E5 and E6: compute time flat from 8 to 12 cores at about 0.10 s) and expect CG to scale further because it does more arithmetic per byte (E6: 70 percent efficiency at 12 cores).

The caveats are these. Every timing is from one laptop with 12 fast and 4 slow cores, so nothing above 12 processes measures the algorithm. The 16-core points measure the hardware (E6, E7, E8). The largest process count tested is 16, against the 256 the rubric requires. There are no cluster numbers and no GPU numbers yet. The RMA result is specific to Open MPI 5 over shared memory on macOS and may not hold on an InfiniBand cluster. The CUDA code has only been validated for correctness through the CPU emulation layer (E3) and has never been compiled by `nvcc` or timed on a GPU. The error analysis and the agreement tests, on the other hand, are complete and do not depend on the machine.

**How would I explain it to a classmate in two sentences?** Use non-blocking halo exchange, compact blocks that keep the contiguous direction whole, a convergence check every 10 to 100 sweeps, and a few multi-threaded MPI processes per node rather than one per core. Everything about speed above 12 cores still has to be measured on the cluster, and the GPU code has only been checked for correctness, not speed.

## E16. What the cluster runs will add

**What is it?** A map from the job scripts in `scripts/atlas/` and `scripts/vanda/` to the figures they will regenerate and the questions they will answer.

**Why did we do it?** So that the reader of this handbook knows which pages are final and which will be replaced after the cluster jobs finish, and so that whoever runs the jobs knows what to look for.

**How does it work?** `scripts/atlas/gen_jobs.sh` writes one single-node job, `node_1.pbs`, holding the serial baselines, every rank count up to the node size, the design study, the hybrid splits of one node and the satellite runs, and one `mpi_NP.pbs` for each larger rank count in `RANKS` (192 on Atlas, 128 and 256 on Vanda), the largest of which also carries the design study, the process-grid shapes and the data accuracy study, plus a `hybrid_NP.pbs` for the splits across the nodes. `scripts/vanda/gpu.pbs` runs the MPI+CUDA binary on one GPU and the MPI+OpenMP binary on the same node's cores, on both problems, and `gpu_multi.pbs` runs two GPUs. They append to `results/cluster/*.csv` with the same columns as the local files, so `make figures RESULTS=results/cluster FIGDIR=results/figures_cluster` produces the same figures plus two new ones, `gpu_vs_cpu.png` and a multi-GPU table, with no changes to the script.

**What came out of it?** Nothing yet. The folder `results/cluster` does not exist until the jobs are run. The questions each figure will answer are these. The strong-scaling figure (`mpi_*.pbs`) will show whether CG's 70 percent efficiency at 12 cores survives to 256 cores on an 8192 by 8192 grid, and at what core count the Jacobi sweep becomes message-latency bound rather than bandwidth bound. The communication-share figure will separate the two communication costs properly: the halo exchange, whose volume per process shrinks with p, against the all-reduce, whose latency grows like log p and which CG pays twice per iteration. On the laptop the two were masked by waiting. The weak-scaling figure will be a genuine network measurement, because each node brings its own memory bandwidth (E8). The exchange-mode figure (`design_*.pbs`) will say whether `rma` behaves on a network with hardware remote memory access, and whether `overlap` pays off once messages cross a real network with higher latency than shared memory. The decomposition figure at 256 processes (1x1x256 against 256x1x1 against 8x8x4) will test whether face contiguity still beats volume when the blocks are 512 by 512 by 2 slabs. The hybrid figure (`hybrid_*.pbs`) will answer the lecture 9 question for a 72-core node directly. The GPU figure (`gpu.pbs`) will show the update rate of one GPU against one full CPU node for the same grids, and `gpu_multi.pbs` whether host-staged or CUDA-aware halo exchange is faster between two GPUs.

**How would I explain it to a classmate in two sentences?** The cluster jobs produce the same CSV columns, so the same `make figures` redraws every figure with 256-core and GPU data and nothing in the analysis changes. Those runs are what will say whether the laptop's lessons about bandwidth, latency and the rank-thread split hold at scale.

## E17. Figure sst_maps.png: one cloudy day, observed and filled

**What is it?** Four maps of the seas around India on 10 April 2025 (day 99 of the file), drawn by `scripts/local/maps.sh` and `analysis/maps.gp`: the temperature as the VIIRS instrument saw it, with white where cloud or land blocked the view; the 2D fill computed from that day alone; the same day cut out of the 3D fill of the whole year; and the difference between the two fills on the filled pixels only. Colours run from 24 degrees (blue) to 32 degrees (red); the difference map runs from -2 (blue) to +2 (red).

**Why did we do it?** Numbers in a table say how large the error is; a map says where the method is believable and where it is not. The day was chosen because it is cloudy enough to be a real test (about two thirds of the sea is missing) but not so cloudy that nothing is left to anchor the fill.

**How does it work?** `gridtool slice` writes each field as a text matrix, with `NaN` for pixels that are missing or land, and gnuplot's `matrix with image` paints every number as a coloured square and leaves `NaN` blank. The difference map is the 3D fill minus the 2D fill, and an `awk` line blanks the observed pixels so that only the filled ones show, which is where the two methods can differ at all.

**What came out of it?** On 10 April 2025 the Bay of Bengal is almost entirely under cloud and the Arabian Sea is clear in the north-west and patchy elsewhere; 65 percent of the sea pixels are missing. The 2D fill is complete and faithful near every cloud edge, but the middle of the Bay of Bengal, hundreds of kilometres from the nearest clear pixel, is a featureless blend of the rim at 29 to 30 degrees. The 3D fill keeps the filaments of the Arabian Sea eddy field running under the cloud and gives the Bay of Bengal a warmer western side and a cooler north-eastern corner, which are what the neighbouring days saw there. The two fills differ by about 0.19 degrees root-mean-square over the filled pixels (`gridtool diff` reports 0.153 over all 285 187 sea pixels of the day, the observed ones being identical in both), nearly zero near observations and up to a degree in the centres of the big holes.

**How would I explain it to a classmate in two sentences?** The top-left map is full of holes, the other two are complete, and the 3D one keeps the shapes of fronts and eddies under the clouds where the 2D one smears them into a blur. The difference map shows the two fills agreeing near the edges of every hole and disagreeing in the middle of the large ones.

## E18. Figure gapfill_accuracy.png and the accuracy table

**What is it?** The left panel plots the root-mean-square error of the fill on hidden pixels against the side of the hidden patches (1, 4, 16 and 64 pixels, that is 4 to 250 km), for the 2D fill of the day, the 3D fill of the year measured on the same day and on all days, next to the baseline of filling every hole with the mean observed temperature. The right panel plots the 3D error against the weight given to the time neighbours (0.01 to 3) for 16-pixel patches, measured on the 2D day and over the year, with the 2D fill of that day as a horizontal reference. The table behind it is "Accuracy of the fill on hidden pixels" in `tables.md`, produced by `do_gapfill()` from the `acc2d` and `acc3d` rows of `gapfill.csv`; it also lists the mean absolute error, the maximum error, the number of hidden pixels, the root-mean-square differences against the reference analysis on filled and on observed pixels, the iteration count and the time.

**Why did we do it?** Hiding single pixels flatters any method, because a single hidden pixel is surrounded by observed ones and its average is almost always right. Real cloud gaps are tens to hundreds of kilometres across, so the honest question is how the error grows with the size of the hole, and whether knowing the neighbouring days helps when the hole is large. The time-weight panel asks how strongly the day before and after should be trusted.

**How does it work?** `--holdout 0.1 --holdout-block B` hides one tenth of the observed pixels in B by B patches, the solver fills them as if they were cloud, and the error is measured against the hidden truth (D4a). The baseline needs no solver. The reference columns compare both the filled and the observed pixels with NOAA's blended analysis, so a reader can see whether our fill disagrees with NOAA more than the satellite itself does.

**What came out of it?** With 16-pixel patches the 2D fill of the day has a root-mean-square error of 0.1546 degrees and a mean absolute error of 0.1074 against a baseline of 0.9223; for 1, 4 and 64-pixel patches the RMSE is 0.0892, 0.1273 and 0.3754. The 3D fill of the year with equal weights, measured on the same hidden pixels of the same day, gives 0.1179, 0.1942, 0.2131 and 0.2854 for the four patch sizes: worse than 2D up to 16 pixels, better at 64. Measured over all 365 days it gives 0.1340, 0.1912, 0.2318 and 0.2483 against a baseline of 1.47 (the baseline is higher over the year because one mean temperature fits a year worse than it fits a day). Lowering the time weight from 1 to 0.3, 0.1, 0.03 and 0.01 lowers the 3D error on the day for 16-pixel patches from 0.2131 to 0.2005, 0.1861, 0.1695 and 0.1595, approaching but never beating the 2D value of 0.1546, while the iteration count rises from 199 to 261, 328, 403 and 499 and the 16-core time from 19 to 49 seconds. Against the blended analysis, the raw VIIRS pixels of the day differ by 0.377 degrees root-mean-square and our filled pixels by 0.26 (2D) and 0.24 to 0.25 (3D); over the year the observed pixels differ by 0.259 and the 3D fill by 0.26 to 0.29.

**How would I explain it to a classmate in two sentences?** Filling a 64 km cloud hole from the clear pixels around it gets the temperature right to 0.15 degrees, six times better than guessing the mean, and the error grows with the size of the hole. Adding yesterday and tomorrow helps only for holes wider than about 100 km and only if the days are given a small weight, because the sea changes more from day to day than from pixel to pixel.

## E19. Figure gapfill_scaling.png and the data scaling table

**What is it?** Strong scaling of the solver on the real data: the 2D day (801 by 534 pixels) and the 3D year (801 by 534 by 365) with CG run to a relative residual of 1e-6 and Jacobi run for a fixed 1000 (2D) or 200 (3D) sweeps, on 1 to 16 ranks, with the speedup measured against the serial binary. The table "Strong scaling on the satellite data" in `tables.md` adds the iteration counts, the imbalance ratio, the communication share and, in its last column, the hidden-pixel error of each run.

**Why did we do it?** The manufactured problem has uniform work everywhere; the data has land, clear sky and cloud in different places, so some ranks have far fewer unknowns than others. This figure says whether that irregularity costs scaling, and the last column of the table is the agreement test for the data problem: every rank count must report the same error.

**How does it work?** `do_gapfill()` treats the `sst*` rows exactly like `do_strong()` treats the manufactured rows: minimum of the repeats, speedup against the serial row, efficiency, communication share.

**What came out of it?** The 2D day runs in 0.585 s serially with CG (489 iterations) and reaches speedups of 5.97 on 8 ranks (efficiency 0.75), 7.12 on 12 (0.59) and 3.84 on 16, the same shape as the manufactured CG case; Jacobi with 1000 sweeps runs in 0.35 s serially and reaches 4.55 and 6.60 on 8 and 12 ranks. The 3D year takes 97.9 s serially with CG (199 iterations) and 37.1 s for 200 Jacobi sweeps; CG reaches 4.89 on 8 ranks (0.61), 4.57 on 12 and 4.69 on 16, with the best 16-core time, 18.7 s, from 4 ranks of 4 threads, against 19.6 s for 16 OpenMP threads and 20.9 s for 16 pure ranks. The imbalance column is the new feature: 1.40 to 1.51 for Jacobi in 2D from 4 ranks upwards and 1.28 to 1.39 in 3D, against 1.00 on the manufactured problem, because the masked sweep costs little on land and observed pixels and much on unknowns, and the unknowns are concentrated under the cloud; CG stays at 1.07 to 1.09. The last column reads 0.1546 for every 2D row and 0.2318 for every 3D row, which is the agreement test. The masked Jacobi sweep processes 0.84 billion cells per second on one core (37.1 s for 200 sweeps of 156 million cells) against 2.1 billion for the plain kernel.

**How would I explain it to a classmate in two sentences?** The real data scales like the textbook problem, about six to seven times faster on 8 to 12 cores, and every rank count reports exactly the same fill error, which is how we know the parallel version is right. The one new effect is that cloud and land are not spread evenly over the tiles, so the Jacobi ranks with the most unknowns finish up to 50 percent later than the average and the others wait.
# Part F. Presenting and defending

This Part is the toolkit for the day someone asks you about the project. F1 is a glossary of every technical term that appears anywhere in the code, the scripts, the report or this handbook, one plain sentence each. F2 is thirty questions an examiner or classmate is likely to ask, with model answers that quote the real numbers from `results/figures/tables.md`. F3 is a ten-minute talk mapped to the sections of the report. F4 is what to do when a number looks wrong: how to trace any figure back through the analysis script, the CSV file and the shell command that produced it.

Two reminders before you use any of this. The timings quoted here come from one Apple M4 Max laptop with 12 performance cores and 4 efficiency cores. The 256-core and GPU runs on the NUS clusters have not been done yet. The scripts for them exist in `scripts/atlas/` and `scripts/vanda/`, and the answers below say so wherever it matters.

## F1. Glossary

Each entry is one sentence. Terms in the same family are grouped under the first word a reader is likely to look up, and the alphabetical order is by that word.

**all-reduce**: an MPI operation in which every process contributes a value, the values are combined (here summed or maximised), and every process receives the combined result.

**Amdahl's law**: the rule that if a fraction S of a program cannot be parallelised, the speedup on n cores is at most 1/(S + (1-S)/n), so it can never exceed 1/S.

**API**: an application programming interface, the published set of functions a library offers to programs that use it.

**atomicAdd**: a GPU instruction that adds a value to a memory location in one indivisible step, so many threads can add to the same location without losing updates.

**bandwidth**: the rate at which data can be moved, measured in bytes per second, whether between memory and processor or between two computers.

**barrier**: a synchronisation point (`MPI_Barrier`) that no process may pass until every process has reached it. The code uses one before starting its timers.

**block (CUDA)**: a group of GPU threads (here 64 x 4 x 1 = 256 threads) that is scheduled together and can share fast memory.

**block (decomposition)**: the rectangular piece of the global grid that one MPI rank owns. Also used for the "block" shape of a process grid as opposed to slabs or pencils.

**block partition**: the way a row of n points is cut into P contiguous pieces so that pieces differ in length by at most one point. Implemented by `partition()` in `src/common.hpp`.

**boundary condition**: the rule for what the solution does at the edge of the domain. Here the Dirichlet condition u = 0 on every edge.

**cache**: a small, fast memory next to the processor that holds recently used data so the processor does not have to wait for main memory.

**Cartesian communicator**: an MPI communicator (`MPI_Cart_create`) in which the processes are arranged in a rectangular grid, so each one can ask who its left, right, up and down neighbours are.

**CG**: conjugate gradient, an iterative method for symmetric positive definite linear systems that chooses search directions which never undo earlier progress.

**coalesced access**: a GPU memory pattern in which consecutive threads read consecutive addresses, which the hardware can serve in one wide transaction.

**collective**: an MPI operation that all processes in a communicator take part in at once, such as a broadcast, a reduce or an all-reduce.

**communicator**: an MPI object naming a group of processes that may talk to each other. `MPI_COMM_WORLD` holds every process, and the code builds a Cartesian one from it.

**compiler**: the program that turns source code into a machine executable. This project was built with Apple clang 21 on the laptop and expects g++ or the cluster's compiler on Linux.

**condition number**: for a matrix, the ratio of its largest to smallest eigenvalue. For the discrete Laplacian it grows like 4/(pi^2 h^2), and CG's iteration count grows like its square root.

**convergence**: the behaviour of an iteration whose answers approach a limit. In the CSV, "converged" means the relative residual dropped below the tolerance before the iteration cap.

**CSV**: comma-separated values, a plain text table with one row per line and commas between columns, written by every solver run with `--csv`.

**CUDA**: Nvidia's programming model and toolkit for running code on its GPUs.

**CUDA-aware MPI**: an MPI implementation that accepts pointers to GPU memory directly, so a program need not copy data to the CPU before sending it.

**.dat file**: a plain text data file with one measurement per line and whitespace between columns, written by `bin/summarize` and read by gnuplot.

**datatype (MPI derived)**: a description of a memory layout that MPI can pack and unpack by itself, such as the non-contiguous face of a 3D block built by `MPI_Type_create_subarray`.

**deadlock**: a situation in which two or more processes each wait for the other to act first, so none ever proceeds. Blocking sends and receives in the wrong order cause it.

**Dirichlet**: the kind of boundary condition that fixes the value of the solution on the boundary (here zero), as opposed to fixing its slope.

**discretisation**: replacing a continuous problem on a region by a finite set of unknowns on a grid and algebraic equations linking them.

**domain decomposition**: splitting the spatial domain into pieces, giving each piece to a process, and exchanging boundary data between pieces.

**dot product**: the sum of the products of corresponding entries of two vectors, written `dot()` in `src/kernels.hpp`.

**double precision**: the 64-bit floating-point number format, with about 16 significant decimal digits, used for every array in this project.

**efficiency (parallel)**: speedup divided by the number of cores, so 1.0 means every added core delivered its full share.

**efficiency core**: one of the four slower, low-power cores on the Apple M4 Max. A process placed on one runs at a fraction of the speed of a performance core.

**eigenvector**: a vector that a matrix merely rescales rather than rotates. The plain sine product is one for the discrete Laplacian, which is why it had to be replaced as the test solution.

**emulation**: running code meant for one kind of hardware on another by imitation. `src/cuda/cuda_compat.hpp` runs CUDA kernels as ordinary CPU loops for testing.

**environment variable**: a named setting in the shell that programs can read, such as `OMP_NUM_THREADS`, which tells OpenMP how many threads to use.

**ERDDAP**: NOAA's data server that returns a slice of a gridded dataset, chosen by writing the ranges of latitude, longitude and time into the web address, as a plain text file.

**finite differences**: approximating derivatives by differences of neighbouring grid values, for example u'' at a point by (u_left - 2u + u_right)/h^2.

**flop**: one floating-point operation (an addition or multiplication of decimal numbers). A Jacobi sweep does 2d + 2 of them per grid point.

**fork-join**: the pattern in which one thread splits into many for a parallel loop and they merge back into one afterwards. Each OpenMP `parallel for` is one fork-join.

**FUNNELED**: the MPI thread-support level (`MPI_THREAD_FUNNELED`) in which a program may have several threads but only the main thread calls MPI.

**gnuplot**: a command-line plotting program that reads columns of numbers from text files and writes image files; `analysis/plots.gp` is its script here.

**GPU**: a graphics processing unit, a processor with thousands of simple cores designed for doing the same operation on large arrays at once.

**grid-stride loop**: a GPU loop in which each thread handles elements t, t + T, t + 2T, ... where T is the total number of threads, so any array size is covered by a fixed number of threads.

**Gustafson's law**: the rule for weak scaling that if the parallel part of the work grows with the core count n while the serial part stays fixed, the scaled speedup is S + (1-S) n.

**halo**: the one-cell-thick layer of copies of a neighbouring rank's boundary values that surrounds each rank's block, refreshed before every sweep.

**harmonic in-painting**: filling a hole in an image or a field with the solution of the Laplace equation, so that every filled pixel is the average of its neighbours and the result is the smoothest field that matches the rim of the hole; the method this project uses on the satellite maps.

**header file**: a C++ source file (ending `.hpp` or `.cuh`) that is included by other files and here carries most of the code, so one copy serves every executable.

**hold-out**: pixels that were observed but are deliberately hidden from the solver (`--holdout`, in patches of `--holdout-block` pixels) so that the filled values can be compared with the truth; the same idea as a test set in machine learning.

**host and device**: in CUDA, the CPU and its memory (host) versus the GPU and its memory (device).

**imbalance**: in the result tables, the slowest rank's compute time divided by the mean over ranks. 1.0 is perfect balance.

**inline**: a C++ keyword asking the compiler to substitute a small function's body at the call site, and allowing the same function to be defined in a header included by many files.

**iteration**: one pass of an iterative method. One Jacobi sweep or one CG step.

**Jacobi**: the iteration that replaces each grid value by the weighted average of its neighbours and the source, using only values from the previous sweep.

**kernel**: a function that runs on the GPU, launched over many threads at once. Also used loosely for the inner loops of the CPU code in `src/kernels.hpp`.

**Krylov space**: the span of f, Af, A^2 f and so on. CG finds the best answer within it, which is why an eigenvector right-hand side is solved in one step.

**L2 norm**: the square root of the sum of squared entries. The discrete L2 error multiplies that sum by the cell volume hx hy hz before the square root.

**Laplacian**: the sum of second derivatives of a function in each spatial direction, written as a triangle (Delta). The Poisson equation says minus the Laplacian equals the source.

**latency**: the fixed delay before any data arrives, independent of how much is sent. A message of one byte still costs the latency.

**least squares**: the fit that chooses the straight line minimising the sum of squared vertical distances to the data points; `slope()` in `summarize.cpp` uses it on logarithms to measure a power-law exponent.

**load balance**: how evenly the work is divided among processes. The block partition guarantees at most one extra layer of points per rank.

**log-log plot**: a chart whose two axes are logarithmic, on which a power law y = c x^k appears as a straight line of slope k.

**land mask**: the byte per pixel that says whether a pixel is sea or land; taken from NOAA's blended analysis and resampled to the 4 km grid, it tells the masked stencil which neighbours to ignore.

**make and Makefile**: the standard build tool and its recipe file. `make all` compiles the four CPU executables.

**manufactured solution**: a test method in which the answer is chosen first and the source term is computed from it, so the solver's error can be measured exactly.

**max norm**: the largest absolute entry of a vector, reported as `linf_err` in the CSV.

**memory-bound**: a computation whose speed is limited by how fast data can be fetched from memory rather than by arithmetic. Jacobi is one.

**MPI**: the Message Passing Interface, the standard library for programs made of separate processes that communicate by sending messages.

**MPI_Allreduce**: the MPI function that performs an all-reduce. The code calls it with one double at a time.

**MPI_Cart_create**: the MPI function that builds a Cartesian communicator from a list of grid dimensions.

**MPI_Dims_create**: the MPI function that chooses a balanced process grid, for example 4 x 4 for 16 ranks, for the dimensions the user left as zero.

**MPI_Isend and MPI_Irecv**: non-blocking send and receive. They return immediately and the transfer completes later, which `MPI_Waitall` waits for.

**MPI_Put**: the one-sided MPI function that writes data into another process's exposed memory window without that process calling receive.

**MPI_Sendrecv**: a combined blocking send and receive that cannot deadlock when every process calls it with matching partners.

**MPI_Type_create_subarray**: the MPI function that builds a derived datatype describing a rectangular sub-block of a multidimensional array.

**MPI_Waitall**: the MPI function that blocks until a list of non-blocking operations has completed.

**MPI_Win**: a window, the MPI object that exposes a region of a process's memory for one-sided access by others.

**mask (pixel)**: the array of one byte per grid cell in the data problem, with codes 0 missing, 1 observed, 2 land, 3 hidden, which the masked kernels read to decide which cells to update and which neighbours to use.

**NaN**: "not a number", the floating-point value that the grid file uses for a missing pixel; any arithmetic with it stays NaN, which is why the mask, not the value, drives the kernels.

**node**: one computer in a cluster, with its own processors and memory. A Vanda CPU node has 72 cores.

**non-blocking**: a communication call that returns before the transfer is done, letting the program do other work in the meantime.

**OpenMP**: the directive-based model for shared-memory multithreading, in which `#pragma omp` lines tell the compiler which loops to run on several threads.

**overlap**: hiding communication behind computation. The `overlap` exchange mode updates the deep interior while halo messages are in flight.

**PBS**: the Portable Batch System, the job scheduler on the NUS clusters, driven by `#PBS` lines at the top of a job script and the `qsub` command.

**pencil**: a process-grid shape in 3D in which the grid is cut along two of the three directions, so each rank owns a long thin column.

**performance core**: one of the twelve fast cores on the Apple M4 Max.

**pinned memory**: host memory that the operating system promises not to move, which lets the GPU copy to and from it at full speed (`cudaMallocHost`).

**Poisson equation**: the partial differential equation minus the Laplacian of u equals f, describing steady heat flow, electrostatics and membranes.

**pragma**: a `#pragma` line in C or C++ that gives the compiler an instruction outside the language proper, such as an OpenMP directive.

**preconditioner**: an approximate inverse applied inside CG to reduce the condition number and the iteration count. This project uses none.

**process grid**: the arrangement of MPI ranks as a PX x PY x PZ grid, printed in the `dims` column of the CSV.

**qsub**: the PBS command that submits a job script to the queue. `qsub -I` asks for an interactive session on a compute node.

**race condition**: a bug in which the result depends on which thread happens to reach shared data first. Jacobi avoids it by writing to a separate array.

**rank**: the integer identity of one MPI process within a communicator, from 0 to size minus 1.

**reduction**: combining many values into one with an operation like sum or max, either across OpenMP threads (`reduction(+:sum)`) or across MPI ranks.

**residual**: the vector r = f - A u that measures how far u is from solving the system. Its relative norm is the stopping criterion.

**RMA**: remote memory access, the one-sided MPI communication model built on windows, `MPI_Put` and fences.

**roundoff**: the tiny error introduced each time a floating-point result is rounded to 16 digits. It explains why CG dot products differ in the last digits across rank counts.

**scheduling (static)**: the OpenMP policy that divides loop iterations into equal contiguous chunks at the start of the loop, appropriate when every iteration costs the same.

**sea-surface temperature (SST)**: the temperature of the top layer of the ocean, measured from space by infrared radiometers (blocked by cloud) and microwave radiometers (coarser, but see through cloud); the quantity our maps show, in degrees Celsius.

**second order**: an error that shrinks by a factor of four each time the grid spacing is halved, because it is proportional to h^2.

**serial**: running on one core with no parallelism. `bin/poisson_serial` is the baseline for every speedup.

**SIMD**: single instruction, multiple data, the processor feature that applies one arithmetic operation to several numbers at once. `#pragma omp simd` asks for it on the inner loop.

**slab**: a process-grid shape in which the grid is cut along only one direction, so each rank owns a thick slice with at most two neighbours.

**speedup**: the serial time divided by the parallel time. Ideal speedup on p cores is p.

**stencil**: the fixed pattern of neighbours used to update one grid point. Five points in 2D, seven in 3D.

**stride**: the distance in memory between consecutive elements along one array dimension. `sy` and `sz` in `Local` are the strides along y and z.

**strong scaling**: holding the problem size fixed and adding cores, measuring how much faster the same job finishes.

**subarray**: a rectangular sub-block of a larger array, the thing `MPI_Type_create_subarray` describes.

**summary.txt**: the file of `key: value` headline numbers (convergence orders, slopes, speedups, best strategies) that `bin/summarize` writes next to the figures.

**symmetric positive definite**: a matrix equal to its transpose whose quadratic form is always positive. The property CG requires and the discrete Laplacian has.

**tag**: an integer label on an MPI message that lets a receiver pick the right one among several. The code uses 10 + d and 20 + d for the two directions in dimension d.

**thread**: a stream of instructions sharing memory with other threads in the same process. OpenMP creates them, MPI processes do not share memory.

**tolerance**: the threshold on the relative residual below which the iteration stops (`--tol`). A value of 0 means run a fixed number of iterations.

**VIIRS**: the Visible Infrared Imaging Radiometer Suite, the instrument on the NOAA-20 (and other) satellites whose infrared channels give the 4 km sea-surface temperature maps used here.

**walltime**: the real elapsed time a job may run, requested in the PBS header and measured by `MPI_Wtime` inside the code.

**weak scaling**: growing the problem in proportion to the core count so every core keeps the same work, measuring whether the time stays flat.

**window**: see MPI_Win.

**zero-flux (Neumann) boundary**: a boundary across which nothing flows, so the field's slope there is zero; in the data problem the coast is treated this way by simply dropping the land neighbours from a pixel's average.

## F2. Thirty likely questions, with model answers

**Q1. Why the Poisson equation? Is it not a toy problem?**
It is the simplest problem with the structure that matters: every unknown is coupled to its neighbours, so no part of the grid can be solved in isolation. The same stencil, halo exchange and global reductions appear inside fluid solvers, electrostatics codes and image processing. It also has known exact solutions, so correctness can be checked to machine precision rather than argued. The lecturer named it explicitly in the project feedback and asked for both the 2D and 3D versions, which we provide from one source tree.

**Q2. The assignment forbids embarrassingly parallel algorithms. Why does this qualify?**
An embarrassingly parallel job can be split into pieces that never talk. Here a Jacobi sweep on one rank needs the current values of the neighbouring rank's edge cells, so every sweep begins with a halo exchange, and CG additionally needs two global sums per iteration. At 16 ranks on the laptop that communication became 70 to 85 percent of the runtime in the strong-scaling table, which is as far from embarrassingly parallel as it gets.

**Q3. Why does the exact solution have the factor exp(x)?**
The first version used u = sin(pi x) sin(pi y). That function is an eigenvector of the discrete Laplacian: the matrix just rescales it. Its right-hand side is then a multiple of the same vector, the initial residual spans a one-dimensional Krylov space, and CG found the exact answer in one iteration on every grid. The timing of a one-iteration solve measures nothing. Multiplying by exp(x) keeps the zero boundary and the smoothness but spreads the solution across many eigenvectors. CG then needs 93, 187, 371, 738 and 1467 iterations on the 2D grids n = 64 to 1024, which is the expected linear growth.

**Q4. How did you validate that the parallel code is correct?**
Three ways, all recorded in `results/local/validation.csv`. First, the discretisation error against the exact solution falls by a factor of 3.88 to 3.99 per grid doubling in 2D and 3.58 to 3.94 in 3D, matching second-order theory, with a measured order of 2.00. Second, Jacobi run to a tolerance of 1e-8 stops at the same sweep in serial and in parallel: 15700 sweeps on the 64 x 64 grid for one rank and for four ranks, 2350 on the 24^3 grid for one and eight ranks, with identical residuals. Third, a fixed 300-sweep Jacobi problem was run through the serial, OpenMP, MPI (all four exchange modes), MPI+OpenMP and emulated CUDA builds, and every one reports an L2 error of 8.3708880689e-01 to all eleven printed digits. The largest deviation across the whole table is 3.2e-15, which is roundoff from summing in a different order.

**Q5. Why does Jacobi need a number of sweeps proportional to n squared?**
Each sweep damps the slowest error mode by a factor cos(pi h), which is about 1 - pi^2 h^2 / 2. To reduce the error by a factor 10^10 therefore takes roughly 2 ln(10^10) / (pi^2 h^2) sweeps, and since h = 1/(n+1) that is proportional to n^2. The measured counts to a tolerance of 1e-10 were 5050 sweeps at n = 32, 19650 at n = 64 and 77300 at n = 128: each doubling of n multiplies the count by about 3.9.

**Q6. And why does CG need only a number proportional to n?**
CG's iteration count grows like the square root of the condition number, and the condition number of the discrete Laplacian grows like 1/h^2, so the count grows like 1/h, which is proportional to n. The 2D counts 93, 187, 371, 738, 1467 for n = 64, 128, 256, 512, 1024 double with n, exactly as predicted. In 3D the counts were 22, 53, 113, 228 for n = 16 to 128.

**Q7. What exactly is a halo?**
Each rank owns a block of the grid, but the stencil at a block's edge cell needs values that belong to the neighbouring rank. So each rank stores an extra one-cell-thick border around its block, the halo, holding copies of the neighbour's edge values. Before every sweep the halos are refreshed by exchanging those edge layers. The halo cells on the physical boundary are never written and stay zero, which implements the Dirichlet condition for free.

**Q8. You implemented four halo-exchange strategies. Which won?**
Non-blocking `isend` was the default and the reference. At 8 ranks the differences between `sendrecv`, `isend` and `overlap` were within a few percent: for Jacobi 2D, 0.119, 0.114 and 0.106 seconds. At 16 ranks `overlap` was the best for the 2D cases (0.90 of the `isend` time for Jacobi, 0.63 for CG) while `sendrecv` fell to 1.95 times `isend` for Jacobi 2D. One-sided `rma` was the clear loser everywhere, from 2.5 to 7 times slower, with one 29-second outlier on Jacobi 3D at 16 ranks.

**Q9. Why was the one-sided RMA mode so slow?**
The implementation uses `MPI_Win_fence` to open and close an access epoch around the puts, and a fence is a collective synchronisation of every rank in the window, twice per sweep. Open MPI on macOS has no hardware support for one-sided operations over shared memory, so the puts are emulated with ordinary messages plus the two fences, which is strictly more work than the `isend` path. The 29-second outlier at 16 ranks, where halo time was 28.99 of 28.98 seconds total, looks like a progress or scheduling pathology with four ranks on efficiency cores, and it is noted as such rather than explained. On a cluster with InfiniBand, where RMA maps onto hardware, the comparison could come out differently, which is one reason the `exchange.csv` block is in the cluster job scripts.

**Q10. Why did overlap help in 2D but hurt in 3D?**
Overlap pays for its hidden communication with extra work: the outer layer of the block is updated afterwards as six separate slabs, and the strided slab loops are less cache-friendly than one pass over the whole block. In 2D the boundary is a small fraction of the block and only four slabs are needed, so the saving on waiting outweighed the extra passes (0.94 of `isend` at 8 ranks, 0.90 at 16). In 3D the surface-to-volume ratio is worse, there are six slabs, and at 8 ranks the compute time rose from 0.119 to 0.150 seconds, more than the 0.063 seconds of halo time it could possibly hide, so total time went from 0.176 to 0.214 seconds.

**Q11. Why were 16 ranks slower than 12 on the laptop?**
The M4 Max has 12 performance cores and 4 efficiency cores. With 16 ranks, four processes land on efficiency cores and run markedly slower. The `imbalance` column, which is the slowest rank's compute time over the mean, jumps from 1.03 at 12 ranks to 1.46 at 16 for Jacobi 2D. Every halo exchange and every all-reduce then waits for the slow ranks, so the time attributed to communication explodes: Jacobi 2D halo time went from 0.016 to 0.150 seconds and CG 2D all-reduce time from 0.070 to 0.610 seconds. The best strong-scaling point on this machine is 12 ranks, with CG 2D at 8.34 times speedup and 70 percent efficiency.

**Q12. What does the imbalance column measure, and why report it?**
It is `t_compute_max / t_compute_mean` across ranks. The code records each rank's own compute, halo and all-reduce time, then reduces the maximum and the mean. If every rank did equal work at equal speed the ratio is 1.0. A value above 1 means some ranks finished late, and since the others can only wait for them inside an MPI call, their waiting shows up as halo or all-reduce time. Reporting the ratio separates true communication cost from waiting caused by uneven speed, which is exactly the distinction needed to interpret the 16-rank results.

**Q13. Jacobi stops scaling at about 8 ranks even without efficiency cores. Why?**
Jacobi is memory-bound: each sweep does about six flops per point but streams three arrays through memory, so one core already moves data at roughly 40 gigabytes per second (1.7 to 2.3 billion updates per second times about 24 bytes). Several cores on the same chip share one memory system, and once it saturates, extra cores add nothing. The pure-OpenMP thread scaling shows the same ceiling without any MPI: 16 threads gave a speedup of only 5.1 for Jacobi 2D and 5.3 for 3D. CG does more arithmetic per byte and scales further, reaching 6.8 times at 8 ranks and 8.3 at 12.

**Q14. What is the difference between strong and weak scaling?**
Strong scaling fixes the problem and adds cores, asking how much faster the same job runs. The ideal is time falling as 1/p. Weak scaling grows the problem with the cores so every rank keeps the same amount of work, asking whether the time stays flat. The ideal is constant time. Strong scaling eventually fails for any problem because each rank's share becomes too small to amortise communication. Weak scaling is the fairer test of whether a code can use a big machine on a correspondingly big problem, and it is the one the lecturer said synthetic data makes easy, because the grid size follows `--local-n` times the process grid.

**Q15. What are speedup and efficiency exactly, and what numbers did you get?**
Speedup is the serial time divided by the parallel time. Efficiency is speedup divided by the core count. In the strong-scaling table on 8 ranks, CG 2D on a 1024^2 grid gave speedup 6.82 and efficiency 0.85, CG 3D on 128^3 gave 5.95 and 0.74, Jacobi 2D gave 4.88 and 0.61, Jacobi 3D gave 3.52 and 0.44. The weak-scaling efficiency at 8 ranks was 0.76 for CG 2D, 0.83 for CG 3D, 0.49 for Jacobi 2D and 0.58 for Jacobi 3D. Speedups above the core count at 2 ranks (1.17 efficiency for Jacobi 2D) come from each rank's half-grid fitting better in cache.

**Q16. How do Amdahl's and Gustafson's laws apply here?**
Amdahl's law says speedup is bounded by 1/(S + (1-S)/n) where S is the serial fraction. Our solvers have almost no serial fraction in the loop itself, so the Amdahl limit is loose. What limits strong scaling is communication whose cost per iteration does not shrink with p, which behaves like a serial fraction that grows with p. Gustafson's law for weak scaling says the scaled speedup is S + (1-S) n, nearly linear when S is small, which is what the weak-scaling efficiencies above 0.8 up to 8 ranks for CG show, until the efficiency cores spoil the 16-rank point.

**Q17. Why do the axes for time, problem size and core count use a log scale?**
Because the quantities span orders of magnitude and the relationships are power laws. On log axes a power law is a straight line whose slope is the exponent: the serial time per iteration has fitted slope 1.05 (Jacobi 2D), 1.01 (Jacobi 3D), 1.01 (CG 2D) and 1.00 (CG 3D) against the number of grid points, which says linear cost at a glance. Ideal speedup is a straight line of slope one, so deviations from it are easy to see. The lecturer asked for log axes on all three kinds of axis in the project notes.

**Q18. Why take the minimum over repeated runs rather than the mean?**
Every disturbance to a timing (another process waking, a page fault, a frequency change) can only add time. Nothing can make a run faster than its intrinsic cost. The minimum is therefore the least noisy estimate of that intrinsic cost. Every configuration was run twice in `run_all.sh` and `bin/summarize` keeps the minimum with the `best()` function. With more repeats the median would also be reasonable. The raw rows are all kept in the CSVs and the workbook so anyone can recompute either.

**Q19. What does `--check-every` do and what did it cost?**
Jacobi has no natural residual, so to know when to stop the code must compute r = f - A u and its global norm, which needs a fresh halo exchange, one extra sweep and one all-reduce. `--check-every C` does that every C sweeps. A value of 0 means never, which is what the fixed-200-sweep benchmarks use. Checking every sweep on 16 ranks raised the Jacobi 2D time from 0.216 to 0.566 seconds (all-reduce time alone 0.345 seconds). Checking every 10 sweeps cost nothing measurable (0.205 seconds), so 10 is the default when a tolerance is given.

**Q20. What was the best split of 16 cores between MPI ranks and OpenMP threads, and why?**
Eight ranks with two threads each was best for all four cases: Jacobi 2D 0.114 seconds against 0.214 for 16 x 1 and 0.126 for 1 x 16. CG 2D 0.780 against 1.186 and 1.022. Pure MPI at 16 x 1 suffered the efficiency-core waiting described in Q11. Pure OpenMP at 1 x 16 had no communication at all but paid fork-join overhead on every loop and also had threads on slow cores. The 8 x 2 split halved the number of ranks that can be slow, kept each rank's block larger, and dropped the communication share from about 75 percent to 20 to 50 percent. This is the question lecture 9 posed about 72-core Vanda nodes, and the cluster hybrid jobs sweep the splits from 256 x 1 to 4 x 64 to answer it there.

**Q21. Why MPI_THREAD_FUNNELED rather than MULTIPLE?**
FUNNELED promises that only the main thread calls MPI, which is true here because every MPI call sits outside the OpenMP parallel regions. It is cheaper than MULTIPLE, which would make the MPI library lock around every call to protect against concurrent use. The price is that threads cannot help with communication, for example by having one thread drive the halo exchange while the others compute. That would be a reasonable next experiment.

**Q22. How does the CUDA version work, and what has been tested?**
Each MPI rank keeps its block in GPU memory. The Jacobi sweep and the operator application are kernels launched over a 3D grid of 64 x 4 x 1 thread blocks, with the contiguous x direction mapped to consecutive threads so memory access is coalesced. Dot products use a grid-stride loop over 1024 x 256 threads with one `atomicAdd` per thread, and one double per rank goes to `MPI_Allreduce`. Faces are packed into contiguous device buffers by a kernel, copied to pinned host memory, sent with non-blocking MPI, copied back and unpacked. The driver has been compiled only through the CPU emulation header and run that way. In emulation it reproduces the serial results to all printed digits for Jacobi and CG, 2D and 3D, staged and CUDA-aware paths. It has not been compiled with nvcc or run on a GPU. The Vanda job `scripts/vanda/gpu.pbs` does that.

**Q23. What is CUDA-aware MPI and does your code use it?**
A CUDA-aware MPI library accepts device pointers in `MPI_Isend` and `MPI_Irecv` and moves the data itself, possibly straight between GPUs with GPUDirect, so the program skips the copy to host memory. The driver has a `--cuda-aware` flag that passes the device buffers directly. Without it, data is staged through pinned host buffers. In emulation both paths give identical answers. Whether the cluster's MPI is CUDA-aware is unknown until tested, and the multi-GPU job tries it and falls back with a message if it fails.

**Q24. How is a 2D problem stored as 3D without wasting time?**
Every block is a 3D array with a one-cell halo on each side, indexed as k * sz + j * sy + i with i contiguous. For 2D the z extent is one, so there is one interior plane and two halo planes in z that are never written. The z coefficient 1/hz^2 is set to zero, so the stencil's z terms contribute nothing, and because the z loop is the outermost it runs once with no inner-loop penalty. The result is one set of kernels and one exchange routine for both dimensions, and the 2D process grid is simply forced to have one layer in z.

**Q25. Why use MPI derived datatypes instead of packing faces by hand?**
A face of a 3D block is not contiguous in memory except for z faces, so sending it needs either a copy into a buffer or a description of the layout. `MPI_Type_create_subarray` provides the description: `face_type()` builds eight datatypes per rank (send and receive, low and high, per dimension) once, and every send or receive then names the array pointer and the datatype. The library does the packing, often more efficiently than a user loop, and the code has no buffer management. The CUDA version packs by hand because the datatypes would have to be applied to device memory, which only a CUDA-aware MPI can do.

**Q26. How do you know the serial baseline and the parallel code do the same work?**
They are the same source. The kernels in `src/kernels.hpp` and the solver loop in `src/solver.hpp` are compiled into `poisson_serial`, `poisson_omp`, `poisson_mpi` and `poisson_mpi_omp`. Only a small policy object differs, and in the serial build its exchange and reduction functions are empty. That is why the fixed-sweep Jacobi results agree to every printed digit across variants: the arithmetic is literally identical and only the order of summation in the diagnostics differs. A baseline written separately could be faster or slower for unrelated reasons and would make the speedups unreliable.

**Q27. What do you expect to change at 256 cores on the cluster?**
Several things, and the job scripts are designed to show them. Cluster nodes have uniform cores, so the imbalance effect of Q11 should disappear and the strong-scaling curves should extend much further. Messages between nodes cross a network with higher latency than shared memory, so the halo and all-reduce times per iteration should rise, and the all-reduce cost should grow like log of the rank count, making CG's two global sums per iteration the dominant communication at large p. The problem sizes were enlarged (8192^2 and 512^3 for Jacobi, 4096^2 and 256^3 for CG) so that each of 256 ranks still owns a quarter-million points. Until those jobs run, these are expectations, not results.

**Q28. What is the single biggest limitation of the study as it stands?**
The absence of the cluster and GPU measurements. Everything about design and correctness is complete and verified, but the 10-point rubric item for scaling to at least 256 cores and the GPU test required by the OpenMP choice are satisfied only by scripts, not by figures. The second limitation is scientific: the laptop's mixed cores mean the 16-rank points measure the scheduler rather than the algorithm, which is why the 12-rank points were added and why the report labels the 16-rank results as a hardware artefact.

**Q29. What would you do next?**
Four things in order of value. Run the cluster and GPU jobs, since they are ready. Add a preconditioner to CG (Jacobi or Chebyshev polynomial), which cuts iterations without adding communication. Try pipelined CG, which rearranges the algorithm so the two all-reduces per iteration become one and can overlap the matrix application. And test a two-cell halo with two sweeps per exchange, which halves the number of messages at the price of a little redundant arithmetic. Each of these targets a cost that the timers in this code can already measure separately.

**Q30. How was AI used, and how do you know the results are real?**
Claude was used as a programming and writing assistant: it drafted the source files, scripts and documents from a design discussed in conversation, ran the local experiments, and drafted the report and this handbook. Every number in the report comes from running the code. The CSV rows, the log and the derived `.dat` files are in the repository so any value can be traced to the command that produced it (Part F4 shows how). Two decisions came out of reviewing the assistant's first draft and are recorded in the report: the exp(x) factor in the test solution after CG converged in one step, and the CPU emulation layer for the CUDA code. The report's AI-usage section says all this in the words the assignment asks for.

**Q31. Why satellite sea-surface temperature, and why around India rather than Singapore?**
Because the cloud problem is real and the method is exactly our equation. Infrared radiometers such as VIIRS cannot see the sea through cloud, so daily maps have holes, and the simplest standard way to fill them is harmonic in-painting, which is the Laplace equation with the observed pixels as fixed values. Singapore was the first idea, but the island is about a dozen 4 km pixels across, far too small to need even two processes. The box around India (5 N to 25 N, 65 E to 95 E) has 801 by 534 pixels, a third of them land, the Arabian Sea and the Bay of Bengal, and a monsoon that covers most of the sea for months, so the holes are large and the problem is worth parallelising. The data are free from NOAA CoastWatch's ERDDAP server, which needs no account; the NASA server we planned to use first was retired in September 2026.

**Q32. How does a hole in a map become a linear system?**
Ask for the smoothest field that agrees with the clear pixels: minimise the sum of squared differences between neighbouring sea pixels with the observed ones held fixed. Setting the derivative to zero gives, for each missing pixel, "my value equals the weighted average of my sea neighbours", which is the Laplace stencil. Land neighbours are simply dropped, so the coast is an insulated wall. In 3D the two neighbours in time are added with weight `--tweight`. Writing the missing pixels as unknowns gives a symmetric positive definite system whose right-hand side holds the observed neighbours, so Jacobi and CG apply unchanged. The code solves for the anomaly around the mean observed temperature, so a missing region with no observed rim at all simply gets the mean.

**Q33. How do you know the fill is any good if you cannot see under the clouds?**
By hiding pixels we can see. `--holdout 0.1 --holdout-block 16` hides one tenth of the observed pixels in 16 by 16 patches (64 km squares, the size of real cloud gaps), fills them as if they were cloud, and compares with the truth. On 10 April 2025, filled from the rest of that day, the hidden 64 km squares come back with a root-mean-square error of 0.15 degrees and a mean absolute error of 0.11, single pixels to 0.09 and 250 km squares to 0.38, against 0.92 degrees for filling every hole with the mean. The baseline of filling every hole with the mean temperature is the number to beat, and we also compare with NOAA's gap-free blended analysis, which is an independent product, not the truth.

**Q34. Does the third dimension (time) actually help?**
Only for large holes, and only with a small weight. On the same hidden pixels of the same day, the year-long fill with equal space and time weights is worse than the one-day fill for patches up to 64 km (0.21 against 0.15 degrees for 16-pixel patches) and better for 250 km patches (0.29 against 0.38). Lowering the time weight to 0.01 brings the 16-pixel error to 0.16, level with 2D, at two and a half times the iterations. The reason is that the sea surface changes by a few tenths of a degree from one day to the next (diurnal warming, cloud-biased sampling, sensor noise), so for a small hole the clear pixels a few kilometres away are better witnesses than the same place a day earlier, while for a hole wider than the spatial coherence of the field the days are the only witnesses. The maps show the second effect: the 3D fill keeps eddies and fronts under the cloud that the 2D fill smooths into a blur.

**Q35. What changes in the parallel code for the real data?**
Very little, by design. The grid is cut the same way (longitude, latitude, day), each rank reads its own tile plus a one-cell rim straight from the grid file, and the filled field is written back with one collective MPI-IO call. The stencil kernels get a masked variant that skips observed and land cells and rebuilds the diagonal from the neighbours that exist; the dot products, vector updates and halo exchange are untouched because fixed cells hold zero in the solver arrays. The hidden pixels are chosen by a hash of their global position, so every rank count hides the same pixels and must report the same error, which is the agreement test. What does change is the work balance: a tile that is mostly land or clear sky has few unknowns, and the imbalance column of the data scaling table shows how much that costs.

**Q36. What would an oceanographer say about your fill?**
That it is the right baseline and not the final word. Harmonic in-painting flattens any front or eddy that lies entirely under a cloud, because inside a hole the solution is a smooth blend of the rim; operational products use optimal interpolation with a covariance model, or DINEOF, which learns the dominant spatial patterns from the clear days and uses them to fill the cloudy ones. Our 3D fill moves one step in that direction by using the neighbouring days, and the natural next steps are weights that depend on how fast the sea changes locally and a rim-aware decomposition so that ranks have equal numbers of unknowns.

## F3. A ten-minute presentation script, mapped to the report

The script below runs about 1500 spoken words, which is ten minutes at a brisk pace. Time stamps are in square brackets. Each passage names the report section it draws on, so slides can be made from that section's figures.

[0:00] **Report Section 2.1: the problem.**
Good afternoon. This project is about solving one equation very fast on many processors, and about what gets in the way. The equation is Poisson's: minus the Laplacian of u equals f. If you have ever wondered what temperature a metal plate settles to when its edges are held at fixed temperatures, or what shape a soap film takes over a wire frame, you have wondered about this equation. We solve it on the unit square and the unit cube with the solution pinned to zero on the boundary.

On a computer we cover the square with a grid. At each grid point the equation becomes one simple rule: a weighted sum of the point and its four neighbours, six in three dimensions, equals the source there. A 4096 by 4096 grid is nearly seventeen million unknowns, and the matrix that links them is never written down. We apply it on the fly.

[1:10] To know whether our answer is right we cheat in the mathematician's way: we choose the answer first, e to the x times sine pi x sine pi y, and compute the source that produces it. One detail turned out to matter. Our first choice, a plain product of sines, is an eigenvector of the matrix. Conjugate gradient solved it in one step, every time, on every grid, and all our timings were measuring nothing. The exponential factor spreads the solution over many eigenvectors and gives a realistic problem.

[1:50] **Report Sections 2.2 and 2.3: two iterations.**
We solve the system two ways. Jacobi replaces every point by the average of its neighbours and the source, over and over. It is beautifully simple and parallel, and painfully slow to converge: about four n squared sweeps, so 77 thousand sweeps on a 128 by 128 grid. We use it as a stencil benchmark with a fixed 200 sweeps.

Conjugate gradient is cleverer. It picks search directions that never undo earlier progress and needs a number of iterations proportional to n, not n squared: 93 iterations at n equals 64, doubling with n up to 1467 at n equals 1024. The price is two global sums per iteration, which in parallel means two all-reduces, two moments when every process must agree.

[2:50] The table in this box also carries our first validation. The error against the exact solution falls by a factor of four every time the grid spacing halves, from 1.36 times ten to the minus four down to 5.5 times ten to the minus seven. That is exactly second-order accuracy.

[3:10] **Report Section 3: parallel design.**
Now the parallel part. We cut the grid into blocks, one per MPI process, arranged in a Cartesian process grid. Each block carries a one-cell border of copies of its neighbours' edge values, the halo, refreshed before every sweep. That exchange is the heart of the project, so we built it four ways: blocking Sendrecv calls, nonblocking sends, the same nonblocking sends with the interior computed while messages fly, and puts straight into memory windows. The faces are described by MPI subarray datatypes, so there is no manual packing on the CPU.

On top of MPI we add OpenMP threads inside each process, and separately a CUDA version where each process drives one GPU. And a point I want to stress: serial, OpenMP, MPI and MPI plus OpenMP are compiled from one source. The fixed-sweep Jacobi results agree across all of them to seventeen digits. The baseline does exactly the arithmetic the parallel code does.

[4:20] **Report Section 5: serial cost.**
Before scaling, the serial code. Time per iteration against grid points on log-log axes is a straight line of slope one in all four cases: Jacobi and CG, 2D and 3D. Jacobi runs at about two billion point updates per second on one core, CG at about four hundred million because each iteration touches six arrays. Total CG time grows faster than linear, slope 1.5 in 2D, because the iteration count grows with n as well.

[4:55] **Report Section 6.1: strong scaling.**
Fixed problem, more cores, on a sixteen-core laptop. Up to eight ranks the story is good: CG in 2D reaches a speedup of 6.8 at 85 percent efficiency, CG in 3D 6.0 at 74 percent. At twelve ranks CG 2D is still climbing, 8.3 times faster. Jacobi flattens earlier, at about five times, because it is memory-bound: it streams three arrays per sweep and one core already uses roughly forty gigabytes per second of bandwidth. Pure OpenMP shows the same ceiling, so it is the memory system, not MPI.

[5:50] Then sixteen ranks, and everything gets worse. The reason is the hardware. This laptop has twelve fast cores and four slow efficiency cores. With sixteen processes, four of them run slow, and the other twelve wait for them at every exchange. Our timers separate compute from waiting, and the imbalance ratio, slowest rank over mean, jumps from 1.03 to 1.46. The communication share jumps to over 80 percent. That is a scheduler artefact, not a property of the algorithm, and it is why cluster runs on uniform cores are the next step.

[6:40] **Report Section 6.2: weak scaling.**
Weak scaling keeps the work per rank constant and grows the grid. The ideal is a flat line. CG stays near flat to eight ranks, efficiency 0.76 in 2D and 0.83 in 3D, and 0.67 at twelve ranks in 3D. Jacobi drops to about 0.5 by eight ranks, the same bandwidth ceiling as before. And again sixteen ranks collapses for the same reason.

[7:15] **Report Sections 6.4 to 7.2: design choices measured.**
Here are the design experiments. First, the exchange modes. At eight ranks the three message-passing variants are within a few percent of each other. At sixteen ranks, overlapping the interior computation with the messages is best in 2D, 0.63 of the baseline time for CG, but it hurts in 3D, because updating the outer layer afterwards as six separate slabs costs more than the waiting it hides. One-sided RMA with fences was three to seven times slower on this platform and had one 29-second outlier. Open MPI emulates one-sided operations over shared memory with messages plus two collective fences per sweep.

[8:10] Second, the hybrid split. We held the cores at sixteen and varied ranks times threads. Eight ranks with two threads each won in all four cases, roughly half the time of sixteen pure MPI ranks. Pure OpenMP, one rank with sixteen threads, had no communication at all but paid fork-join overhead on every loop. The lecture asked how to split a 72-core node. On this laptop the answer is a few threads per rank, and the cluster jobs sweep from 256 by 1 down to 4 by 64 to answer it there.

Third, the residual check. Checking Jacobi's residual every sweep costs an extra exchange, an extra sweep and an all-reduce, and it more than doubled the runtime. Checking every ten sweeps cost nothing measurable.

[8:40] **Report Section 8: the satellite data.**
Finally the real problem. Infrared satellites cannot see the sea through cloud, so every daily temperature map has holes, and around India in the monsoon the holes cover most of the map. We fill them with the same solver: each cloudy pixel becomes the average of its sea neighbours, the clear pixels are held fixed, and the coast is an insulated wall. In 2D we fill one day from itself; in 3D we stack the whole of 2025, 156 million pixels, and let yesterday and tomorrow help. We test it by hiding patches of pixels we did know. Filled from the day alone, 64-kilometre hidden squares come back to within 0.15 degrees, six times better than guessing the mean. Stacking the year made small holes slightly worse, because the sea changes more from day to day than from pixel to pixel, and large holes better, 0.29 against 0.38 degrees for 250-kilometre squares, and it restored the eddies under the clouds that the one-day fill smooths away. The solver scaled on the real field as on the textbook one, six to seven times faster on 8 to 12 cores, with one new effect: tiles under cloud carry more unknowns than clear tiles, and the Jacobi ranks finish up to 50 percent apart.

[9:15] **Report Section 9: what we learned, and what is still to run.**
So what did we learn. Correctness first: the parallel code is verified to roundoff in every variant, including the CUDA code through a CPU emulation layer. The halo exchange is cheap in volume and expensive in latency: under one percent of the data, but four to six messages per sweep. The global sums in CG are the cost that grows with the machine. Memory bandwidth, not communication, caps Jacobi on one node. And heterogeneous cores can look like a communication problem unless you measure imbalance separately.

[9:40] And honestly, what is still to run. The 256-core strong and weak scaling on Atlas or Vanda, with the design study repeated at 64 and 256 ranks, and the GPU runs comparing one GPU with one CPU node: the job scripts are written and the analysis script produces the cluster figures from the same CSV format. Until they are run, the laptop figures are what we have, and the report says so. Thank you. I am happy to take questions.

## F4. If a number looks wrong: tracing any result back to its input

Every number in the report and this handbook comes from a chain with four links, and each link is a file you can open. The chain is: a shell command in `scripts/local/run_all.sh` runs a binary with `--csv` and `--tag`. The binary appends one row to a CSV in `results/local/`. A `do_*` function in `analysis/summarize.cpp` (compiled to `bin/summarize`) reads that CSV, keeps the minimum time per configuration and computes derived columns, writing a table into `results/figures/tables.md` and a `.dat` file. gnuplot (`analysis/plots.gp`) draws the figure PNG from the `.dat` file. To check any number, walk the chain backwards.

### The procedure

First, identify the figure or table and find its producer. Each figure name maps to one function in `summarize.cpp`: `serial_scaling.png` comes from `do_serial`, `strong_scaling.png` and `strong_comm_fraction.png` from `do_strong`, `weak_scaling.png` from `do_weak`, `exchange_modes.png` from `do_exchange`, `decomposition_shape.png` from `do_decomp`, `check_interval.png` from `do_checkevery`, `hybrid_split.png` from `do_hybrid`, `openmp_threads.png` from `do_omp`, and the validation tables from `do_validation`. The `main()` function at the bottom of `summarize.cpp` lists them with the CSV each one loads, and the matching plot block in `plots.gp` is the one whose `set output` names the PNG.

Second, open the CSV and find the rows. Filter by the `tag` column (`jac2d`, `jac3d`, `cg2d`, `cg3d`, or `order2d`, `agree`, `jacobi_conv` in `validation.csv`), then by `ranks`, `threads`, `exchange`, `dims` or `check_every` as needed. There will be two rows per configuration because `run_all.sh` repeats each run twice. The analysis uses the one with the smaller `t_total`.

Third, find the command. Search `run_all.sh` for the tag and the stage. The helpers `run_serial`, `run_omp`, `run_mpi` and `run_hybrid` show exactly which binary ran with which arguments, and the `--csv` path tells you which file the row went to.

Fourth, rerun that one command by hand, without `--csv`, and compare the printed `t_total` and errors with the CSV row. Timings will differ by a few percent between runs. Errors and iteration counts must match exactly for Jacobi with fixed sweeps, and to about twelve digits for CG.

### Trace 1: the CG 2D speedup of 8.34 at 12 ranks

The strong-scaling figure shows CG 2D reaching a speedup of 8.34 at 12 cores. In `summarize.cpp`, `do_strong` computes `double sp = t1 / r.t_total`, where `t1` is the `t_total` of the row with `impl == "serial"` for the same tag and `r` runs over the non-serial rows, and writes `sp` as the third column of `strong_cg2d.dat`; `plots.gp` then plots that column against cores with `u 1:3` in the speedup panel. So two rows of `results/local/strong.csv` are involved, both with `tag` equal to `cg2d`.

The serial rows are:

```
serial,2,1024,1024,1,cg,isend,1,1,1x1x1,1458,1,7.235544e-09,...,3.887888,...,cg2d
serial,2,1024,1024,1,cg,isend,1,1,1x1x1,1458,1,7.235544e-09,...,3.864801,...,cg2d
```

and `best()` keeps the second, 3.864801 seconds. The 12-rank rows are:

```
mpi,2,1024,1024,1,cg,isend,12,1,4x3x1,1458,1,7.234376e-09,...,0.463227,...,cg2d
mpi,2,1024,1024,1,cg,isend,12,1,4x3x1,1458,1,7.234376e-09,...,0.465950,...,cg2d
```

and the minimum is 0.463227. The speedup is 3.864801 / 0.463227 = 8.343, printed as 8.34, and the efficiency is 8.343 / 12 = 0.695, printed as 0.70. Both runs took 1458 iterations with the same L2 error to seven digits (5.4567092e-07 serial, 5.4567107e-07 on 12 ranks, the difference in the eighth digit being the different summation order of the dot products), so the comparison is like for like.

The commands that produced these rows are in the strong-scaling stage of `run_all.sh`:

```bash
run_serial "$CSV" cg2d  --dim 2 --n 1024 --solver cg --tol 1e-8 --iters 1000000
run_mpi "$CSV" cg2d  $np --dim 2 --n 1024 --solver cg --tol 1e-8 --iters 1000000
```

with `np` taking the values in `RANKS`, which included 12 because the rerun was launched with `EXTRA_RANKS=12`. To reproduce the pair by hand:

```bash
bin/poisson_serial --dim 2 --n 1024 --solver cg --tol 1e-8 --iters 1000000
mpirun -np 12 bin/poisson_mpi --dim 2 --n 1024 --solver cg --tol 1e-8 --iters 1000000
```

The same two rows produce the line `12 0.463227 8.34321 0.695268 0.195034` in `results/figures/strong_cg2d.dat`, so the figure, the table and this trace all rest on the same two measurements.

### Trace 2: the 29-second RMA bar

The exchange-mode chart has one bar that dwarfs the others: Jacobi 3D, 16 ranks, `rma`, about 29 seconds. `do_exchange` reads `results/local/exchange.csv`, groups by `tag` and `ranks`, and plots `t_total` and `t_halo_max` for each `exchange` value. The rows are:

```
mpi,3,192,192,192,jacobi,rma,16,1,4x2x2,200,0,9.663107e-01,0.58589...,1.68603...,32.714199,0.016201,0.150738,32.757429,0.018164,0.136548,32.747280,0.014627,0,0,...,jac3d
mpi,3,192,192,192,jacobi,rma,16,1,4x2x2,200,0,9.663107e-01,0.58589...,1.68603...,28.984252,0.013496,0.156398,28.989904,0.019588,0.138286,28.977862,0.015894,0,0,...,jac3d
```

Both repeats are slow (32.7 and 29.0 seconds), so this is not a single hiccup. The `t_compute_max` column is 0.15 seconds, in line with every other mode, and `t_halo_max` is 28.99 seconds with `t_halo_mean` 28.98: every rank spent the whole time inside the exchange. The answer is right (the L2 error 0.5858925 matches the other modes to all digits), so this is purely a performance pathology of the one-sided path. The producing line in `run_all.sh` is in the design stage:

```bash
run_mpi "$CSV" jac3d $np --dim 3 --n 192  --iters 200 --exchange $mode
```

with `mode` equal to `rma` and `np` equal to 16. To reproduce or to probe it, run `mpirun -np 16 bin/poisson_mpi --dim 3 --n 192 --iters 200 --exchange rma` and compare with `--exchange isend`. Try `-np 12` to see whether the efficiency cores are involved, and try `--dims 2,2,4` to see whether the process-grid shape matters. The report treats this as an unexplained outlier of the fence-based RMA implementation on this platform, which is the honest description.

### Trace 3: an l2_err in the agreement table

The agreement table reports an L2 error of 8.3708880689e-01 for Jacobi 2D, n = 300, 300 sweeps, for every implementation. To trace one such value, start from `do_validation`, which selects rows with `tag == "agree"` from `results/local/validation.csv` and prints `l2_err` with ten decimals. The serial row is:

```
serial,2,300,300,1,jacobi,isend,1,1,1x1x1,300,0,9.706536e-01,0.83708880688971243,1.7034276923218474,...,agree
```

The value 0.83708880688971243 was written by `write_csv()` in `src/common.hpp` with the format `%.17g` from `R.l2_err`. That field is filled at the end of `solve()` in `src/solver.hpp`:

```cpp
error_norms(P, L, u.data(), e2, einf);
const double cell = L.hx * L.hy * (P.dim == 3 ? L.hz : 1.0);
R.l2_err = std::sqrt(C.sum(e2) * cell);
```

`error_norms()` in `src/kernels.hpp` loops over the interior, subtracts `exact_at(P, L, L.ox + i, L.oy + j, L.oz + k)` from `u` at each cell and accumulates the squared difference into `e2` and the largest absolute difference into `einf`. `exact_at()` in `src/common.hpp` evaluates exp(x) sin(pi x) sin(pi y) at the global coordinate x = gi * hx, with hx = 1 / (n + 1) = 1 / 301. The global index uses the rank's offset `L.ox`, which is why the same cell gives the same exact value on any rank. `C.sum(e2)` is the identity in the serial build and an `MPI_Allreduce` in the parallel builds, which is where the last-digit differences between implementations come from: adding the same partial sums in a different order rounds differently.

To check the number independently, run `bin/poisson_serial --dim 2 --n 300 --iters 300` and read the `l2_err` it prints, or, for a stronger check, note that this is a 300-sweep Jacobi run far from convergence (the relative residual is still 0.97), so the error is large and dominated by the unconverged iterate, not the discretisation. Running with `--solver cg --tol 1e-9` instead gives 6.3277e-06, which is the discretisation error on that grid.

### Common causes when a number is wrong

**gnuplot not installed.** `make figures` writes the tables and `.dat` files and then fails at the gnuplot step with "command not found". Install it (`brew install gnuplot` on macOS, `module load gnuplot` on the cluster, `apt install gnuplot` on Linux) and run `make figures` again; the tables were already written.

**A stale `bin/summarize`.** After editing `analysis/summarize.cpp`, run `make figures` (or `make summarize`) so that `make` rebuilds the tool; `make` tracks the dependency, but a copy of the binary moved elsewhere will silently run the old code.

**A stale CSV with duplicate rows.** `--csv` appends. Running a stage twice without deleting the CSV gives four rows per configuration instead of two. The minimum rule still works, but a configuration run under different conditions (another job in the background) can win. `run_all.sh` deletes each CSV at the start of its stage. If you run commands by hand, delete or move the CSV first.

**`OMP_NUM_THREADS` unset or wrong.** `poisson_omp` and `poisson_mpi_omp` read the thread count from the environment unless `--threads` is given. With it unset, OpenMP uses every core, so an "8 ranks x 2 threads" run may actually be 8 x 16 and oversubscribe the machine. The `threads` column in the CSV shows what was actually used.

**Oversubscription.** More ranks times threads than cores makes the timings meaningless and inflates halo and all-reduce times. Check `ranks * threads` against the core count, and remember that on this laptop only 12 of the 16 cores are fast.

**A stale binary.** The kernels, the solver and the decomposition live in headers. After editing any `.hpp` file, run `make all` again. `make` tracks the header dependencies listed in `HDRS`, but a binary built before the edit will silently run the old code if `make` is skipped.

**Comparing different workloads.** Speedup must compare the same problem at the same iteration count. Check that `nx`, `ny`, `nz`, `solver`, `iters` and `tol` match between the serial and parallel rows. CG with a tolerance will take the same iteration count on any rank count only because the arithmetic is identical, and a difference of one iteration between rows is a sign that something else changed.

**Timing noise.** A single timing can be off by ten percent on a laptop with other applications open. Compare the two repeats in the CSV. If they disagree by more than that, rerun the configuration with nothing else running.
