import mlrun


def get_data(context, source_url: mlrun.DataItem):
    # Convert the DataItem to a pandas DataFrame
    df = source_url.as_df()

    # Log the DataFrame size after the run
    context.log_result("num_rows", df.shape[0])

    # Store the dataset in your artifacts database
    context.log_dataset("cleaned_data", df=df, index=False, format="csv")
