
####################################################################
# Multi dimensional Gaussian Naïve Bayes
#  Estimate the posterior probability assumoing a Gaussian probability distribution.
# PARAMETERS
#  priors : array-like of shape (n_classes,), default=None
#     Prior probabilities of the classes. If specified, the priors are not adjusted according to the data.
#
#  var_smoothing : float, default=1e-9
#     Portion of the largest variance of all features that is added to variances for calculation stability.
#
# TO BE COMPLETELY COMPATIBLE TO THE NAÏVE BAYES CLASSIFIER, THE FOLLOWING PARAMETERS AND FUNCTIONS ARE REQUIRED.
# m.class_count_              m.epsilon_                  m.get_metadata_routing()    m.partial_fit(              m.predict_log_proba(        m.score(                    m.set_partial_fit_request(  m.var_                      
# m.class_prior_              m.feature_names_in_         m.get_params(               m.predict(                  m.predict_proba(            m.set_fit_request(          m.set_score_request(        m.var_smoothing             
# m.classes_                  m.fit(                      m.n_features_in_            m.predict_joint_log_proba(  m.priors                    m.set_params(               m.theta_ 
#
class MDGaussianNB: 

    def __init__(self, var_smoothing=1e-9, var_smoothing_grid=):
        # The constructor to initialize the attributes
        super().__init__(var_smoothing=var_smoothing)
        # self.var_smoothing_grid = var_smoothing_grid
        # self.optimal_smoothings = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        n_features = X.shape[1]
        self.models = []
        self.optimal_smoothings = []

        for i in range(n_features):
            gnb = GaussianNB()

            # Find var_smoothing for the current feature
            if self.var_smoothing_grid is not None:
                param_grid = {'var_smoothing': self.var_smoothing_grid}
                grid_search = GridSearchCV(gnb, param_grid, cv=5, scoring='accuracy')
                grid_search.fit(X[:, i:i+1], y)
                best_smoothing = grid_search.best_params_['var_smoothing']
            else:
                best_smoothing = self.var_smoothing

            # Train the model with the optimal var_smoothing
            gnb.set_params(var_smoothing=best_smoothing)
            gnb.fit(X[:, i:i+1], y)

            # Save the model and the smoothing value
            self.models.append(gnb)
            self.optimal_smoothings.append(best_smoothing)

        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples = X.shape[0]
        n_classes = len(self.models[0].classes_)
        joint_log_likelihood = np.zeros((n_samples, n_classes))

        for i, model in enumerate(self.models):
            joint_log_likelihood += model.predict_proba(X[:, i:i+1])

        return joint_log_likelihood / len(self.models)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


