#Adjusted R Square
dtrain = lgb.Dataset(data=X, label=y)
      def lgb_adjusted_r2_score(preds, dtrain):
            labels = dtrain.get_label()
            n=dtrain.num_data()
            k=dtrain.num_feature()
            return 'r2', ((1-r2_score(labels, preds))*(n-1))/(n-k-1), True