import { Heart, BookOpen, Database, Cpu, Shield, ExternalLink } from 'lucide-react';
import { Link } from 'react-router-dom';
import results from '../../../outputs/dashboard/results';
import Panel from '../components/Panel';

const details = [
  {
    title: 'Dataset', icon: Database,
    items: ['WESAD: Wearable Stress and Affect Detection', '15 lab subjects; ECG, EDA, TEMP, RESP, ACC', 'Baseline, stress, amusement'],
  },
  {
    title: 'Models', icon: Cpu,
    items: ['Logistic Regression, Random Forest, XGBoost, LightGBM', 'Residual 1D-CNN on raw signals', 'LOSO with training-fold imputation and scaling'],
  },
  {
    title: 'Features', icon: BookOpen,
    items: [`${results.binary.n_features} features; 60 s windows, 50% overlap`, 'HRV, EDA, temperature, respiration, motion', 'EDA tonic/phasic decomposition'],
  },
  {
    title: 'Explainability', icon: Shield,
    items: ['Binary XGBoost mean |SHAP| on the full-data fit', 'Descriptive; no held-out or causal interpretation', 'Motion confounds and dataset-transfer limitations'],
  },
];

const links = [
  ['Repository', 'https://github.com/urmeo/CalmSense'],
  ['Architecture', 'https://github.com/urmeo/CalmSense#architecture'],
  ['References', 'https://github.com/urmeo/CalmSense#references'],
  ['License', 'https://github.com/urmeo/CalmSense/blob/main/LICENSE'],
];

export default function About() {
  return (
    <div className="space-y-8 animate-fade-in max-w-4xl mx-auto">
      <div className="text-center">
        <div className="flex items-center justify-center space-x-3 mb-4">
          <Heart className="w-12 h-12 text-red-500" />
          <h1 className="text-4xl font-bold text-gradient">CalmSense</h1>
        </div>
        <p className="text-xl text-gray-600 dark:text-gray-400">Wearable stress detection · v1.0.0</p>
      </div>

      <Panel title="Project">
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          Signal preprocessing, feature extraction, and Leave-One-Subject-Out evaluation:
          each fold tests a person excluded from training.
          The dashboard displays saved experiment results.
        </p>
        <Link to="/models" className="inline-block mt-4 text-blue-600 dark:text-blue-400 underline">
          Compare model accuracy, macro-F1, and transfer results
        </Link>
      </Panel>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {details.map(({ title, icon: Icon, items }) => (
          <Panel key={title} title={<span className="flex items-center gap-3"><Icon className="w-6 h-6 text-blue-500" />{title}</span>}>
            <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-400">
              {items.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </Panel>
        ))}
      </div>

      <nav aria-label="Project resources" className="flex flex-wrap justify-center gap-4">
        {links.map(([label, href]) => (
          <a key={label} href={href} target="_blank" rel="noopener noreferrer"
            className="flex items-center gap-2 text-blue-600 dark:text-blue-400 underline">
            {label}<ExternalLink aria-hidden="true" className="w-4 h-4" />
          </a>
        ))}
      </nav>
      <p className="text-center text-gray-500 dark:text-gray-400 text-sm">© 2025 Urme Bose · MIT License</p>
    </div>
  );
}
