# Magpie Filter Block

Block used for filtering data enriched via Magpie tagging and distance blocks.
Modified version of [**Magpie**](https://magpie-align.github.io/) to enable working with opensource models.

## Data specification

### Required

```
- `filter_type`: one of the following [all, invalid_samples,invalid_scores,high_quality_filter]
- `hq_filter_criteria` : (Optional) criteria for filtering in terms of 'input_quality','sample_quality' or 'difficulty'. To be provided as a dictionary of list in yaml style . If hq_filter_criteria is not given the default high quality filter criteria used by magpie paper will be used (see explanantion). If hq_filter is given but only difficulty is provided then for the other two criteria, all possible values will be retained as criteria ie it will only filter based on difficulty values provided.
- `remove_duplicates` : True/False whether to perform deduplication based on distance tagging. Default is False
```

### Format of Data

To perform Magpie filtering, the data should also have the following :

```
`input_quality` : output of input quality
`judge_quality_score` : output of sample quality
`difficulty` : output of difficulty tagging
`min_similar_uuid` : output of distance tagging
```

### Explanation

There are 3 existing filters :

invalid_samples : If the data has a conversations field, it will look at the 'user' and 'assistant' roles and make sure there is atleast one of each and their length is greater than 0. Otherwise it will look at the input and output field and remove samples where input or output is None or the length of input or output is 0.

invalid_scores : if the format of the output from the magpie tagging stage is not proper then it filters those samples

high_quality_filter : these are the criteria used by Magpie paper to filter samples

```
`input_quality` in ["good", "excellent"]
`judge_quality_score` in ["5"]
`min_similar_uuid` is None
```
