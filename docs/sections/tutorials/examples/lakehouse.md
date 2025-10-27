# DMF integration

When using the DMF integration, you need to:

1. Add configuration to `.env` file as follows:

   ```yaml
   LAKEHOUSE_TOKEN=<DMF Lakehouse token goes here>
   LAKEHOUSE_ENVIRONMENT=<PROD or STAGING>
   ```

   Follow the instructions [here](https://github.ibm.com/IBM-Research-AI/lakehouse-eval-api#lakehouse-token) to generate your lakehouse token.

2. Install DMF Lakehouse dependencies as follows:

   ```command
   pip install ".[dmf-lakehouse]"
   ```

3. Set the datastore type as dmf and the DMF/Lakehouse namespace where the data will be saved.

   Notice: You must have write access to the namespace.

   ```yaml
        data_builder: ...
        seed_examples: ....
        ...
        datastore:
            type: multi_target
            primary:
               type: default
            additional:
               - type: dmf-lakehouse
                 namespace: <namespace (e.g., digit)>
   ```

The use of `multi_target` datastore in this instance uses a primary datastore (in this case, the `default` stores / loads data locally) to save / load data and then _additionally_ saves the data to each datastore in `additional` (in this case, being the dmf-lakehouse datastore)