package MA.MovieAnalysis;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
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
				.add("Color","string")
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
				.add("num_voted_users","long")
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
				.add("title_year","string")
				.add("actor_2_facebook_likes","long")
				.add("imdb_score","double")
				.add("aspect_ratio","double")
				.add("movie_facebook_likes","long")
				;
		Dataset<Row> training = spark.read()
				.option("header", "true")
				.schema(schema)
				.csv("resources//movie_metadata.csv");
		Dataset<Row> testing = spark.read()
				.option("header", "true")
				.csv("resources//movie_metadata.csv");
		training = training.na().drop();
		Dataset<Row>[] splits = training.randomSplit(new double[] {0.9, 0.1}, 12345);
		DataArray.add(splits[0]);
		DataArray.add(splits[1]);
		return DataArray;
		
	}
	
	public PipelineModel linReg(Dataset<Row> trainingdata)
	{
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[] {"duration","budget"})//Has high correlation
				.setOutputCol("features");
		LinearRegression lr = new LinearRegression()
				.setFeaturesCol("features") //All features
				.setLabelCol("imdb_score") //Target
				.setMaxIter(400)
				.setRegParam(0.01)
				.setElasticNetParam(0.1);
		Pipeline Pipeline = new Pipeline().setStages(new PipelineStage[]{assembler,lr});
		PipelineModel model = Pipeline.fit(trainingdata);
		LinearRegressionModel lrModel = (LinearRegressionModel) model.stages()[1];
		LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
		System.out.print(trainingSummary.rootMeanSquaredError());
		return model;
	}
	
	public static void main(String Args[])
	{	
		Analysis ma = new Analysis();
		List<Dataset<Row>> Datasets = ma.dataFetch();
		Datasets.get(0).show();
		PipelineModel TrainedModel = ma.linReg(Datasets.get(0));
		Dataset<Row> predictions = TrainedModel.transform(Datasets.get(1));
		predictions.select("features", "imdb_score", "prediction").show(1000);
	}
	
}

