# What is DiGiT

DiGiT is a framework which enables different algorithms and models to be used to generate synthetic data. It provides a defined interface that allows a diverse set of algorithms and models to be used in a "Plug and Play" fashion.

- [Get started now](./getting_started/installation.md)
- [View it on GitHub](https://github.ibm.com/DGT/fms-dgt)

## Motivation

Synthetic data generation (SDG) involves the use of computational methods and simulations to create data. This means that seed data is used to create artificial data that have some of the statistical characteristics of the seed data. This is a game changer when fine tuning models using user data contributions as it acts as a "petri dish" to magnify the data for tuning. As SDG can be resource intensive depending on the algorithm and model used, it would be really useful to be able to choose the algorithm and model as per your preference and resource capability.

## Architecture

The key architectural components are:

- Task: A task that will be executed by a data builder. It contains global definitions, termination criteria, and seed data
  - Seed Data: The raw input data to the SDG algorithm
- Data Builder: The algorithm that generates the new synthetic data. Each builder takes a set of tasks which follow its expected format
- Blocks: The main unit of heavy-duty processing in DiGiT. We provide a number of these to help speed up computation at known bottlenecks
  - Generator: Expensive computation that produces some output, most often a Large Language Model (LLM) which is used to generate synthetic data.
  - Validator: Validates the generated synthetic data.

The overall architecture is fairly straightforward. At the top level, there are _tasks_ and _databuilders_. Tasks specify independent data generation objectives. These will have their own seed data, their own termination criteria, their own global variable definitions, etc. Data builders specify the process by which data for a set of tasks should be generated. A data builder will have its own specification for what it expects as inputs. Roughly speaking, there is a many-to-one correspondence between tasks and data builders (though, in theory, tasks could be solved by different data builders as long as their data abided by the constraints of the data builder).
