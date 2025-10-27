# Knowledge SDG

The Knowledge-SDG [pipeline](https://github.ibm.com/conversational-ai/fms-dgt-internal/tree/develop/src/databuilders/generation/knowledge_sdg) is used for generating synthetic data in the form of instruction-response pairs by leveraging external knowledge. The data generated is then used for knowledge-tuning a Large Language Model. The pipeline consists of two stages - textbook qa/ mcq generation and verification. Each stage uses a particular template abd has its own set of principles and instructions that control the role of the teacher model (generator vs evaluator) that help guide the generation/evaluation process.

> **NOTE**
>
> This pipeline is an implementation of the LAB method described in [Large-Scale Alignment for ChatBots](https://arxiv.org/abs/2403.01081). 


## Prerequisites

To successfully run this, you will need to have completed the following:

1. Followed the [installation guide](../getting_started/installation.md) to set up your virtual environment  
2. Read through the [Data Builders](../key_concepts/databuilders.md) and [Task](../key_concepts/tasks.md) sections


## Running Knowledge-SDG out-of-the-box



## Create a new task yaml file

For this exercise, let's create a task file that helps generate data for teaching a model about photosynthesis using external document(s). 

As described in the [Task](../key_concepts/tasks.md) section, let's create a `qna.yaml` file in the following directory.

```shell
$ mkdir data/knowledge/photosynthesis
```

Within this directory, add a `qna.yaml` file with the following lines:

```yaml
# data/knowledge/photosynthesis/qna.yaml
task_name: knowledge_photosynthesis
created_by: IBM Research
task_description: "To teach a language model about photosynthesis"
```

Since we're using the `knowledge_sdg` databuilder, let's add:

```yaml
data_builder: knowledge_sdg
```

Before defining the `seed_examples` let's look at the task [definition](https://github.ibm.com/conversational-ai/fms-dgt-internal/blob/develop/src/databuilders/generation/knowledge_sdg/task.py) of Knowledge-SDG to see if we missed any other fields. The `KnowledgeSdgTask` class doesn't define any additional fields, so we can go ahead with adding the `seed_examples`. 

If we take a look at `KnowledgeSdgData` class, we get an idea of the fields that need to be present inside the `seed_examples`. We note that there are 5 fields:
- `taxonomy_path`: This will be auto-populated during instantiation, so we can skip this
- `task_description`: This will also be auto-populated
- `domain`: This needs to be provided. 
- `question`: This needs to be provided.
- `answer`: This also needs to be provided.
- `document`: This is an optional field depending on the task.

Let's add in the `domain` field:

```yaml
domain: plants
```

Let's now add in some seed examples:

```yaml
seed_examples:
  - answer: The word respiration is commonly used to describe the process of breathing in oxygen and breathing out carbon dioxide.
    question: What is respiration?
  - answer: An ecosystem is a community of organisms and their physical environment interacting together.
    question: What is an ecosystem?
  - answer: Metabolism is the chemical reactions in the body's cells that change food into energy.
    question: What is metabolism?
```

Next, we need to specify the knowledge document for our task. We can do this using the `include` directive. 

```yaml
include:
   documents: 
      photosynthesis: documents/photosynthesis.md
```

Let's also create a directory `documents` and a file `photosynthesis.md` inside `data/knowledge/photosynthesis`

```shell
mkdir data/knowledge/photosynthesis/documents
```

```markdown
What is photosynthesis?

Photosynthesis is arguably the most important biological process on earth. By liberating oxygen and consuming carbon dioxide, it has transformed the world into the hospitable environment we know today. Directly or indirectly, photosynthesis fills all of our food requirements and many of our needs for fiber and building materials. The energy stored in petroleum, natural gas and coal all came from the sun via photosynthesis, as does the energy in firewood, which is a major fuel in many parts of the world. This being the case, scientific research into photosynthesis is vitally important. If we can understand and control the intricacies of the photosynthetic process, we can learn how to increase crop yields of food, fiber, wood, and fuel, and how to better use our lands. The energy-harvesting secrets of plants can be adapted to man-made systems which provide new, efficient ways to collect and use solar energy. These same natural "technologies" can help point the way to the design of new, faster, and more compact computers, and even to new medical breakthroughs. Because photosynthesis helps control the makeup of our atmosphere, understanding photosynthesis is crucial to understanding how carbon dioxide and other "greenhouse gases" affect the global climate. In this document, we will briefly explore each of the areas mentioned above, and illustrate how photosynthesis research is critical to maintaining and improving our quality of life.

Photosynthesis and food. All of our biological energy needs are met by the plant kingdom, either directly or through herbivorous animals. Plants in turn obtain the energy to synthesize foodstuffs via photosynthesis. Although plants draw necessary materials from the soil and water and carbon dioxide from the air, the energy needs of the plant are filled by sunlight. Sunlight is pure energy. However, sunlight itself is not a very useful form of energy; it cannot be eaten, it cannot turn dynamos, and it cannot be stored. To be beneficial, the energy in sunlight must be converted to other forms. This is what photosynthesis is all about. It is the process by which plants change the energy in sunlight to kinds of energy that can be stored for later use. Plants carry out this process in photosynthetic reaction centers. These tiny units are found in leaves, and convert light energy to chemical energy, which is the form used by all living organisms. One of the major energy-harvesting processes in plants involves using the energy of sunlight to convert carbon dioxide from the air into sugars, starches, and other high-energy carbohydrates. Oxygen is released in the process. Later, when the plant needs food, it draws upon the energy stored in these carbohydrates. We do the same. When we eat a plate of spaghetti, our bodies oxidize or "burn" the starch by allowing it to combine with oxygen from the air. This produces carbon dioxide, which we exhale, and the energy we need to survive. Thus, if there is no photosynthesis, there is no food. Indeed, one widely accepted theory explaining the extinction of the dinosaurs suggests that a comet, meteor, or volcano ejected so much material into the atmosphere that the amount of sunlight reaching the earth was severely reduced. This in turn caused the death of many plants and the creatures that depended upon them for energy.

Photosynthesis and energy. One of the carbohydrates resulting from photosynthesis is cellulose, which makes up the bulk of dry wood and other plant material. When we burn wood, we convert the cellulose back to carbon dioxide and release the stored energy as heat. Burning fuel is basically the same oxidation process that occurs in our bodies; it liberates the energy of "stored sunlight" in a useful form, and returns carbon dioxide to the atmosphere. Energy from burning "biomass" is important in many parts of the world. In developing countries, firewood continues to be critical to survival. Ethanol (grain alcohol) produced from sugars and starches by fermentation is a major automobile fuel in Brazil, and is added to gasoline in some parts of the United States to help reduce emissions of harmful pollutants. Ethanol is also readily converted to ethylene, which serves as a feedstock to a large part of the petrochemical industry. It is possible to convert cellulose to sugar, and then into ethanol; various microorganisms carry out this process. It could be commercially important one day.

Our major sources of energy, of course, are coal, oil and natural gas. These materials are all derived from ancient plants and animals, and the energy stored within them is chemical energy that originally came from sunlight through photosynthesis. Thus, most of the energy we use today was originally solar energy!

Photosynthesis, fiber, and materials. Wood, of course, is not only burned, but is an important material for building and many other purposes. Paper, for example, is nearly pure photosynthetically produced cellulose, as is cotton and many other natural fibers. Even wool production depends on photosynthetically-derived energy. In fact, all plant and animal products including many medicines and drugs require energy to produce, and that energy comes ultimately from sunlight via photosynthesis. Many of our other materials needs are filled by plastics and synthetic fibers which are produced from petroleum, and are thus also photosynthetic in origin. Even much of our metal refining depends ultimately on coal or other photosynthetic products. Indeed, it is difficult to name an economically important material or substance whose existence and usefulness is not in some way tied to photosynthesis.

Photosynthesis and the environment. Currently, there is a lot of discussion concerning the possible effects of carbon dioxide and other "greenhouse gases" on the environment. As mentioned above, photosynthesis converts carbon dioxide from the air to carbohydrates and other kinds of "fixed" carbon and releases oxygen to the atmosphere. When we burn firewood, ethanol, or coal, oil and other fossil fuels, oxygen is consumed, and carbon dioxide is released back to the atmosphere. Thus, carbon dioxide which was removed from the atmosphere over millions of years is being replaced very quickly through our consumption of these fuels. The increase in carbon dioxide and related gases is bound to affect our atmosphere. Will this change be large or small, and will it be harmful or beneficial? These questions are being actively studied by many scientists today. The answers will depend strongly on the effect of photosynthesis carried out by land and sea organisms. As photosynthesis consumes carbon dioxide and releases oxygen, it helps counteract the effect of combustion of fossil fuels. The burning of fossil fuels releases not only carbon dioxide, but also hydrocarbons, nitrogen oxides, and other trace materials that pollute the atmosphere and contribute to long-term health and environmental problems. These problems are a consequence of the fact that nature has chosen to implement photosynthesis through conversion of carbon dioxide to energy-rich materials such as carbohydrates. Can the principles of photosynthetic solar energy harvesting be used in some way to produce non-polluting fuels or energy sources? The answer, as we shall see, is yes.
```
Source: https://bioenergy.asu.edu/why-study-photosynthesis

## Generate data

We can now run the generation job

```shell
$ fms_dgt --data-path ./data/knowledge/photosynthesis --num-prompt-instructions 3
```

> **NOTE**
>
> `--num-prompt-instructions` is used to specify the no. of in-context learning examples to use in prompts

The output will be written to `./output/knowledge_photosynthesis/data.jsonl` 

Here's a sample output: 
```json
{"task_name": "knowledge_photosynthesis", "taxonomy_path": "knowledge_photosynthesis", "task_description": "To teach a language model about photosynthesis", "domain": "plants", "question": "What is the role of photosynthesis in food production?", "answer": "Photosynthesis plays an essential role in food production as it is the process by which plants, algae, and some bacteria convert light energy, carbon dioxide, and water into glucose and oxygen. This process supplies food for the plant and releases oxygen, which is essential for the survival of most organisms. Animals, including humans, rely on plants as their primary food source, either by consuming plants directly or by consuming herbivores that have eaten plants. This energy transfer from plants to animals forms the basis of food chains and webs in ecosystems. Photosynthesis is, therefore, critical for food production and the survival of most life forms on Earth.\n", "document": {"content": "What is photosynthesis? Photosynthesis is arguably the most important biological process on earth. By liberating oxygen and consuming carbon dioxide, it has transformed the world into the hospitable environment we know today. Directly or indirectly, photosynthesis fills all of our food requirements and many of our needs for fiber and building materials. The energy stored in petroleum, natural gas and coal all came from the sun via photosynthesis, as does the energy in firewood, which is a major fuel in many parts of the world. This being the case, scientific research into photosynthesis is vitally important. If we can understand and control the intricacies of the photosynthetic process, we can learn how to increase crop yields of food, fiber, wood, and fuel, and how to better use our lands. The energy-harvesting secrets of plants can be adapted to man-made systems which provide new, efficient ways to collect and use solar energy. These same natural \"technologies\" can help point the way to the design of new, faster, and more compact computers, and even to new medical breakthroughs. Because photosynthesis helps control the makeup of our atmosphere, understanding photosynthesis is crucial to understanding how carbon dioxide and other \"greenhouse gases\" affect the global climate. In this document, we will briefly explore each of the areas mentioned above, and illustrate how photosynthesis research is critical to maintaining and improving our quality of life. Photosynthesis and food. All of our biological energy needs are met by the plant kingdom, either directly or through herbivorous animals. Plants in turn obtain the energy to synthesize foodstuffs via photosynthesis. Although plants draw necessary materials from the soil and water and carbon dioxide from the air, the energy needs of the plant are filled by sunlight. Sunlight is pure energy. However, sunlight itself is not a very useful form of energy; it cannot be eaten, it cannot turn dynamos, and it cannot be stored. To be beneficial, the energy in sunlight must be converted to other forms. This is what photosynthesis is all about. It is the process by which plants change the energy in sunlight to kinds of energy that can be stored for later use. Plants carry out this process in photosynthetic reaction centers. These tiny units are found in leaves, and convert light energy to chemical energy, which is the form used by all living organisms. One of the major energy-harvesting processes in plants involves using the energy of sunlight to convert carbon dioxide from the air into sugars, starches, and other high-energy carbohydrates. Oxygen is released in the process. Later, when the plant needs food, it draws upon the energy stored in these carbohydrates. We do the same. When we eat a plate of spaghetti, our bodies oxidize or \"burn\" the starch by allowing it to combine with oxygen from the air. This produces carbon dioxide, which we exhale, and the energy we need to survive. Thus, if there is no photosynthesis, there is no food. Indeed, one widely accepted theory explaining the extinction of the dinosaurs suggests that a comet, meteor, or volcano ejected so much material into the atmosphere that the amount of sunlight reaching the earth was severely reduced. This in turn caused the death of many plants and the creatures that depended upon them for energy. Photosynthesis and energy. One of the carbohydrates resulting from photosynthesis is cellulose, which makes up the bulk of dry wood and other plant material. When we burn wood, we convert the cellulose back to carbon dioxide and release the stored energy as heat. Burning fuel is basically the same oxidation process that occurs in", "path": "knowledge_photosynthesis", "document_len": 600, "domain": "plants"}}
{"task_name": "knowledge_photosynthesis", "taxonomy_path": "knowledge_photosynthesis", "task_description": "To teach a language model about photosynthesis", "domain": "plants", "question": "What is cellulose and how is it related to photosynthesis?", "answer": "Cellulose is a carbohydrate that results from photosynthesis. It is the primary structural component of plant cell walls and is the most abundant organic polymer on Earth. During photosynthesis, plants convert carbon dioxide from the air into glucose, starches, and other high-energy carbohydrates, including cellulose. Cellulose is a complex polysaccharide made up of long chains of glucose molecules. It is indigestible to humans and many other animals, but it is an important energy source for ruminants, such as cows and sheep, which have specialized digestive systems that can break down cellulose. When cellulose is burned as fuel, it is converted back to carbon dioxide and water, releasing the stored energy as heat.\n", "document": {"content": "What is photosynthesis? Photosynthesis is arguably the most important biological process on earth. By liberating oxygen and consuming carbon dioxide, it has transformed the world into the hospitable environment we know today. Directly or indirectly, photosynthesis fills all of our food requirements and many of our needs for fiber and building materials. The energy stored in petroleum, natural gas and coal all came from the sun via photosynthesis, as does the energy in firewood, which is a major fuel in many parts of the world. This being the case, scientific research into photosynthesis is vitally important. If we can understand and control the intricacies of the photosynthetic process, we can learn how to increase crop yields of food, fiber, wood, and fuel, and how to better use our lands. The energy-harvesting secrets of plants can be adapted to man-made systems which provide new, efficient ways to collect and use solar energy. These same natural \"technologies\" can help point the way to the design of new, faster, and more compact computers, and even to new medical breakthroughs. Because photosynthesis helps control the makeup of our atmosphere, understanding photosynthesis is crucial to understanding how carbon dioxide and other \"greenhouse gases\" affect the global climate. In this document, we will briefly explore each of the areas mentioned above, and illustrate how photosynthesis research is critical to maintaining and improving our quality of life. Photosynthesis and food. All of our biological energy needs are met by the plant kingdom, either directly or through herbivorous animals. Plants in turn obtain the energy to synthesize foodstuffs via photosynthesis. Although plants draw necessary materials from the soil and water and carbon dioxide from the air, the energy needs of the plant are filled by sunlight. Sunlight is pure energy. However, sunlight itself is not a very useful form of energy; it cannot be eaten, it cannot turn dynamos, and it cannot be stored. To be beneficial, the energy in sunlight must be converted to other forms. This is what photosynthesis is all about. It is the process by which plants change the energy in sunlight to kinds of energy that can be stored for later use. Plants carry out this process in photosynthetic reaction centers. These tiny units are found in leaves, and convert light energy to chemical energy, which is the form used by all living organisms. One of the major energy-harvesting processes in plants involves using the energy of sunlight to convert carbon dioxide from the air into sugars, starches, and other high-energy carbohydrates. Oxygen is released in the process. Later, when the plant needs food, it draws upon the energy stored in these carbohydrates. We do the same. When we eat a plate of spaghetti, our bodies oxidize or \"burn\" the starch by allowing it to combine with oxygen from the air. This produces carbon dioxide, which we exhale, and the energy we need to survive. Thus, if there is no photosynthesis, there is no food. Indeed, one widely accepted theory explaining the extinction of the dinosaurs suggests that a comet, meteor, or volcano ejected so much material into the atmosphere that the amount of sunlight reaching the earth was severely reduced. This in turn caused the death of many plants and the creatures that depended upon them for energy. Photosynthesis and energy. One of the carbohydrates resulting from photosynthesis is cellulose, which makes up the bulk of dry wood and other plant material. When we burn wood, we convert the cellulose back to carbon dioxide and release the stored energy as heat. Burning fuel is basically the same oxidation process that occurs in", "path": "knowledge_photosynthesis", "document_len": 600, "domain": "plants"}}
```

As you can see, the document gets automatically chunked. To control the size of the document chunks, you can specify the following field in the task yaml: 

```yaml
chunk_size: 800 # no. of tokens per document chunk
```