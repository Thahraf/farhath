import weka.classifiers.Classifier;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Stacking;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.Logistics;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class EnsembleExampleOne
{
	public static void main(String arg[])
	{
		try{
			DataSource source = new DataSource("C:/Users/Farhath/Desktop/New folder/weather.nominal.arff");
			Instances dataset = source.getDataSet();

			dataset.setClassIndex(dataset.numAttributes()-1);
			
			System.out.println("====BAGGING===");
			performBagging(dataset);
	
			System.out.println("====BOOSTING===");
			performBoosting(dataset);

			System.out.println("====STACKING===");
			performStacking(dataset);
		}catch(Exception e){
			e.printStackTrace();
		}
	}

	public static void performBagging(Instances dataset) throws Exception{
		Classifier baseClassifier = new J48();

		Bagging bagger = new Bagging();
		bagger.setClassifier(baseClassifier);
		bagger.setNumIterations(10);

		bagger.buildClassifier(dataset);
		System.out.println(bagger.toString);		
	}

	public static void performBoosting(Instances dataset) throws Exception{
		Classifier baseClassifier = new J48();

		AdaBoostM1 booster = new AdaBoostM1();
		booster.setClassifier(baseClassifier);
		booster.setNumIterations(10);

		booster.buildClassifier(dataset);
		System.out.println(booster.toString);		
	}


	public static void performStacking(Instances dataset) throws Exception{
		Classifier[] classifiers = {
			new J48(),
			new SMO()
		};

		Stacking stacker = new Stacking();
		stacker.setClassifiers(classifiers);
		stacker.setMetaClassifier(new Logistic());

		stacker.buildClassifier(dataset);
		System.out.println(stacker.toString);		
	}
} 