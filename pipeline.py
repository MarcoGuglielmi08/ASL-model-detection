from src.extraction.mediapipe_landmarks_extractor import MediapipeLandmarksExtractor
from src.preprocessing.DataIntegrityCheck import DataIntegrityCheck
from src.preprocessing.FeatureEngineering import FeatureEngineering
from src.preprocessing.GeometricOutlier import GeometricOutlierCheck
from src.preprocessing.StructuralConsistencyCheck import StructuralConsistencyCheck
from src.evalutation.ModelExplainability import ModelExplainability
from src.evalutation.Statistical_test import StatisticalTesting
from src.training.model_nested_kfold_cv import ModelNestedKFold
from src.training.train import Trainer

if __name__ == "__main__":

    print("Running ASL landmark pipeline...")

    MediapipeLandmarksExtractor().run()
    DataIntegrityCheck().run()
    GeometricOutlierCheck().run()
    StructuralConsistencyCheck().run()
    FeatureEngineering().run()

    ModelNestedKFold().run()
    StatisticalTesting().run()
    Trainer().run()
    ModelExplainability().run()

    print("Pipeline complete.")
