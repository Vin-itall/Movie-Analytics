package MA.MovieAnalysis;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SparkSession.Builder;
import org.apache.spark.sql.types.StructType;

public class Analysis {
	SparkSession spark = new Builder().appName("MovieAnalysis").config("spark.master", "local").getOrCreate();
	
	public List<Dataset<Row>> dataFetch()
	{
		List<Dataset<Row>> DataArray = new ArrayList<Dataset<Row>>();
		StructType schema = new StructType()
				.add("color","string")
				.add("director_name","string")
				.add("num_critic_for_reviews","integer")
				.add("duration","integer")
				.add("director_facebook_likes","long")
				.add("actor_3_facebook_likes","long")
				.add("actor_2_name","string")
				.add("actor_1_facebook_likes","long")
				.add("gross","long")
				.add("genres","string")
				.add("actor_1_name","string")
				.add("movie_title","string")
				.add("num_voted_users","integer")
				.add("cast_total_facebook_likes","long")
				.add("actor_3_name","string")
				.add("facenumber_in_poster","integer")
				.add("plot_keywords","string")
				.add("movie_imdb_link","string")
				.add("num_user_for_reviews","long")
				.add("language","string")
				.add("country","string")
				.add("content_rating","string")
				.add("budget","long")
				.add("year","integer")
				.add("actor_2_facebook_likes","long")
				.add("imdb_score","double")
				.add("aspect_ratio","double")
				.add("movie_facebook_likes","long")
				;
		Dataset<Row> training = spark.read()
				.option("mode", "DROPMALFORMED")
				.schema(schema)
				.csv("resources//movie_metadata.csv");
		Dataset<Row> testing = spark.read()
				.option("mode", "DROPMALFORMED")
				.schema(schema)
				.csv("resources//movie_metadata.csv");
		DataArray.add(training);
		DataArray.add(testing);
		return DataArray;
		
	}
	public static void main(String Args[])
	{	
		Analysis ma = new Analysis();
		List<Dataset<Row>> Datasets = ma.dataFetch();
		Datasets.get(0).show();
	}
	
}

