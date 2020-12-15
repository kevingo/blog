# Analog layout automation study & notes

* Why analog layout automation is difficult？

> In real-world, analog circuits contain a large number of variants even for a single functionality. For example, there are more than 100 operational transconductor amplifier (OTA) topologies appeared in the textbook and research papers [1, 5]. Examples include telescopic, folded-cascode, and Miller-compensated. Designers may also develop their customized structures to boost the performance. This makes the automatic annotation challenging and difficult to generalize.

## ALIGN - ALIGN: Analog Layout, Intelligently Generated from Netlists

* In 2017, DARPA announced the Electronic Resurgence Initiative (ERI) and started to sponsor the **ALIGN and MAGICAL** projects for fully automated analog layout generation with “no-human-in-theloop” through its IDEA program.

* Created by: University of Minnesota, Texas A&M University, and Intel Corporation.

* Objective: automatically translate an unannotated (or partially annotated) SPICE netlist of an analog circuit to a GDSII layout.

ALIGN contains the following steps:

1. _**Circuit annotation**_ creates a multilevel hierarchical representation of the input netlist. This representation is used to implement the circuit layout in using a hierarchical manner.

2. _**Design rule abstraction**_ creates a compact **JSON-format represetation of the design rules in a PDK**. This repository provides a mock PDK based on a FinFET technology (where the parameters are based on published data). These design rules are used to guide the layout and ensure DRC-correctness.

3. _**Primitive cell generation**_ works with primitives, i.e., blocks the lowest level of design hierarchy, and generates their layouts. Primitives typically contain a small number of transistor structures (each of which may be implemented using multiple fins and/or fingers). A parameterized instance of a primitive is automatically translated to a GDSII layout in this step.

4. _**Placement and routing**_ performs block assembly of the hierarchical blocks in the netlist and routes connections between these blocks, while obeying a set of analog layout constraints. At the end of this step, the translation of the input SPICE netlist to a GDSII layout is complete.
