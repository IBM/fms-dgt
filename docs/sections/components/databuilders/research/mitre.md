The instruction data is generated using the mitreattack Python library, which provides access to tools and utilities for working with MITRE ATT&CK data. This library allows for the extraction of information related to MITRE objects, including tactics, techniques, groups, software, and more.



The extracted data is mapped to predefined task templates to generate various types of tasks, such as Chain of Thought (CoT), Leave-One-Out, and multiple-choice questions.



Task Types

The data builder supports diverse task types, including:



MITRE Mapping: Mapping different objects within the MITRE ATT&CK framework based on object description.



MITRE Description: Generating descriptive tasks based on MITRE data. Given Mitre object name the model will be asked to provide description.



Chain of Thought (CoT): Encouraging models to think step-by-step through complex relationships between Mitre objects.



Leave-One-Out: The model will be asked .



These diverse task formats are designed to help the Granite model enhance its understanding of the relationships between various entities within MITRE ATT&CK, enabling it to "connect the dots" more effectively.
