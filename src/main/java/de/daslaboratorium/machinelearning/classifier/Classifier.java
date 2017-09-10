package de.daslaboratorium.machinelearning.classifier;

import java.io.Serializable;
import java.util.Collection;
import java.util.Dictionary;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;

/**
 * Abstract base extended by any concrete classifier. It implements the basic
 * functionality for storing categories or features and can be used to calculate
 * basic probabilities – both category and feature probabilities. The classify
 * function has to be implemented by the concrete classifier class.
 *
 * @param <T> A feature class
 * @param <K> A category class
 * @author Philipp Nolte
 */
public abstract class Classifier<T, K> implements IFeatureProbability<T, K>, Serializable {

    /**
     * Generated Serial Version UID (generated for v1.0.7).
     */
    private static final long serialVersionUID = 5504911666956811966L;

    /**
     * Initial capacity of category dictionaries.
     */
    private static final int INITIAL_CATEGORY_DICTIONARY_CAPACITY = 16;

    /**
     * Initial capacity of feature dictionaries. It should be quite big, because
     * the features will quickly outnumber the categories.
     */
    private static final int INITIAL_FEATURE_DICTIONARY_CAPACITY = 32;

    /**
     * The initial memory capacity or how many classifications are memorized.
     */
    private int memoryCapacity = 1000;

    /**
     * A dictionary mapping features to their number of occurrences in each
     * known category.
     */
    private Dictionary<K, Dictionary<T, Integer>> featureCountPerCategory;

    /**
     * A dictionary mapping features to their number of occurrences.
     */
    private Dictionary<T, Integer> totalFeatureCount;

    /**
     * A dictionary mapping categories to their number of occurrences.
     */
    private Dictionary<K, Integer> totalCategoryCount;

    /**
     * The classifier's memory. It will forget old classifications as soon as
     * they become too old.
     */
    private Queue<Classification<T, K>> memoryQueue;

    /**
     * Constructs a new classifier without any trained knowledge.
     */
    protected Classifier() {
        reset();
    }

    /**
     * Resets the <i>learned</i> feature and category counts.
     */
    public void reset() {
        featureCountPerCategory = new Hashtable<>(
                INITIAL_CATEGORY_DICTIONARY_CAPACITY);
        totalFeatureCount = new Hashtable<>(INITIAL_FEATURE_DICTIONARY_CAPACITY);
        totalCategoryCount = new Hashtable<>(INITIAL_CATEGORY_DICTIONARY_CAPACITY);
        memoryQueue = new LinkedList<>();
    }

    /**
     * Returns a <code>Set</code> of features the classifier knows about.
     *
     * @return The <code>Set</code> of features the classifier knows about.
     */
    public Set<T> getFeatures() {
        return ((Hashtable<T, Integer>) totalFeatureCount).keySet();
    }

    /**
     * Returns a <code>Set</code> of categories the classifier knows about.
     *
     * @return The <code>Set</code> of categories the classifier knows about.
     */
    public Set<K> getCategories() {
        return ((Hashtable<K, Integer>) totalCategoryCount).keySet();
    }

    /**
     * Retrieves the total number of categories the classifier knows about.
     *
     * @return The total category count.
     */
    public int getCategoriesTotal() {
        int toReturn = 0;
        for (Enumeration<Integer> e = totalCategoryCount.elements(); e.hasMoreElements(); ) {
            toReturn += e.nextElement();
        }
        return toReturn;
    }

    /**
     * Retrieves the memory's capacity.
     *
     * @return The memory's capacity.
     */
    public int getMemoryCapacity() {
        return memoryCapacity;
    }

    /**
     * Sets the memory's capacity. If the new value is less than the old value,
     * the memory will be truncated accordingly.
     *
     * @param memoryCapacity The new memory capacity.
     */
    public void setMemoryCapacity(int memoryCapacity) {
        for (int i = this.memoryCapacity; i > memoryCapacity; i--) {
            memoryQueue.poll();
        }
        this.memoryCapacity = memoryCapacity;
    }

    /**
     * Increments the count of a given feature in the given category. This is
     * equal to telling the classifier, that this feature has occurred in this
     * category.
     *
     * @param feature  The feature, which count to increase.
     * @param category The category the feature occurred in.
     */
    public void incrementFeature(T feature, K category) {
        Dictionary<T, Integer> features = featureCountPerCategory.get(category);
        if (features == null) {
            featureCountPerCategory.put(category,
                    new Hashtable<>(INITIAL_FEATURE_DICTIONARY_CAPACITY));
            features = featureCountPerCategory.get(category);
        }
        Integer count = features.get(feature);
        if (count == null) {
            features.put(feature, 0);
            count = features.get(feature);
        }
        features.put(feature, ++count);

        Integer totalCount = totalFeatureCount.get(feature);
        if (totalCount == null) {
            totalFeatureCount.put(feature, 0);
            totalCount = totalFeatureCount.get(feature);
        }
        totalFeatureCount.put(feature, ++totalCount);
    }

    /**
     * Increments the count of a given category. This is equal to telling the
     * classifier, that this category has occurred once more.
     *
     * @param category The category, which count to increase.
     */
    public void incrementCategory(K category) {
        Integer count = totalCategoryCount.get(category);
        if (count == null) {
            totalCategoryCount.put(category, 0);
            count = totalCategoryCount.get(category);
        }
        totalCategoryCount.put(category, ++count);
    }

    /**
     * Decrements the count of a given feature in the given category. This is
     * equal to telling the classifier that this feature was classified once in
     * the category.
     *
     * @param feature  The feature to decrement the count for.
     * @param category The category.
     */
    public void decrementFeature(T feature, K category) {
        Dictionary<T, Integer> features = featureCountPerCategory.get(category);
        if (features == null) {
            return;
        }
        Integer count = features.get(feature);
        if (count == null) {
            return;
        }
        if (count == 1) {
            features.remove(feature);
            if (features.isEmpty()) {
                featureCountPerCategory.remove(category);
            }
        } else {
            features.put(feature, --count);
        }

        Integer totalCount = totalFeatureCount.get(feature);
        if (totalCount == null) {
            return;
        }
        if (totalCount == 1) {
            totalFeatureCount.remove(feature);
        } else {
            totalFeatureCount.put(feature, --totalCount);
        }
    }

    /**
     * Decrements the count of a given category. This is equal to telling the
     * classifier, that this category has occurred once less.
     *
     * @param category The category, which count to increase.
     */
    public void decrementCategory(K category) {
        Integer count = totalCategoryCount.get(category);
        if (count == null) {
            return;
        }
        if (count == 1) {
            totalCategoryCount.remove(category);
        } else {
            totalCategoryCount.put(category, --count);
        }
    }

    /**
     * Retrieves the number of occurrences of the given feature in the given
     * category.
     *
     * @param feature  The feature, which count to retrieve.
     * @param category The category, which the feature occurred in.
     * @return The number of occurrences of the feature in the category.
     */
    public int getFeatureCount(T feature, K category) {
        Dictionary<T, Integer> features = featureCountPerCategory.get(category);
        if (features == null) {
            return 0;
        }
        Integer count = features.get(feature);
        return (count == null) ? 0 : count;
    }

    /**
     * Retrieves the total number of occurrences of the given feature.
     *
     * @param feature The feature, which count to retrieve.
     * @return The total number of occurences of the feature.
     */
    public int getFeatureCount(T feature) {
        Integer count = totalFeatureCount.get(feature);
        return (count == null) ? 0 : count;
    }

    /**
     * Retrieves the number of occurrences of the given category.
     *
     * @param category The category, which count should be retrieved.
     * @return The number of occurrences.
     */
    public int getCategoryCount(K category) {
        Integer count = totalCategoryCount.get(category);
        return (count == null) ? 0 : count;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float featureProbability(T feature, K category) {
        final float totalFeatureCount = getFeatureCount(feature);

        if (totalFeatureCount == 0) {
            return 0;
        } else {
            return getFeatureCount(feature, category) / (float) getFeatureCount(feature);
        }
    }

    /**
     * Retrieves the weighed average <code>P(feature|category)</code> with
     * overall weight of <code>1.0</code> and an assumed probability of
     * <code>0.5</code>. The probability defaults to the overall feature
     * probability.
     *
     * @param feature  The feature, which probability to calculate.
     * @param category The category.
     * @return The weighed average probability.
     * @see Classifier#featureProbability(Object,
     * Object)
     * @see Classifier#featureWeighedAverage(Object,
     * Object, IFeatureProbability, float, float)
     */
    public float featureWeighedAverage(T feature, K category) {
        return featureWeighedAverage(feature, category, null, 1.0f, 0.5f);
    }

    /**
     * Retrieves the weighed average <code>P(feature|category)</code> with
     * overall weight of <code>1.0</code>, an assumed probability of
     * <code>0.5</code> and the given object to use for probability calculation.
     *
     * @param feature    The feature, which probability to calculate.
     * @param category   The category.
     * @param calculator The calculating object.
     * @return The weighed average probability.
     * @see Classifier#featureWeighedAverage(Object,
     * Object, IFeatureProbability, float, float)
     */
    public float featureWeighedAverage(T feature, K category, IFeatureProbability<T, K> calculator) {
        return featureWeighedAverage(feature, category, calculator, 1.0f, 0.5f);
    }

    /**
     * Retrieves the weighed average <code>P(feature|category)</code> with the
     * given weight and an assumed probability of <code>0.5</code> and the given
     * object to use for probability calculation.
     *
     * @param feature    The feature, which probability to calculate.
     * @param category   The category.
     * @param calculator The calculating object.
     * @param weight     The feature weight.
     * @return The weighed average probability.
     * @see Classifier#featureWeighedAverage(Object,
     * Object, IFeatureProbability, float, float)
     */
    public float featureWeighedAverage(T feature, K category, IFeatureProbability<T, K> calculator, float weight) {
        return featureWeighedAverage(feature, category, calculator, weight, 0.5f);
    }

    /**
     * Retrieves the weighed average <code>P(feature|category)</code> with the
     * given weight, the given assumed probability and the given object to use
     * for probability calculation.
     *
     * @param feature            The feature, which probability to calculate.
     * @param category           The category.
     * @param calculator         The calculating object.
     * @param weight             The feature weight.
     * @param assumedProbability The assumed probability.
     * @return The weighed average probability.
     */
    public float featureWeighedAverage(T feature, K category, IFeatureProbability<T, K> calculator, float weight,
                                       float assumedProbability) {

        /*
         * use the given calculating object or the default method to calculate
         * the probability that the given feature occurred in the given
         * category.
         */
        final float basicProbability = (calculator == null) ? featureProbability(feature, category)
                                                            : calculator.featureProbability(feature, category);

        Integer totals = totalFeatureCount.get(feature);
        if (totals == null) {
            totals = 0;
        }
        return (weight * assumedProbability + totals * basicProbability) / (weight + totals);
    }

    /**
     * Train the classifier by telling it that the given features resulted in
     * the given category.
     *
     * @param category The category the features belong to.
     * @param features The features that resulted in the given category.
     */
    public void learn(K category, Collection<T> features) {
        learn(new Classification<T, K>(features, category));
    }

    /**
     * Train the classifier by telling it that the given features resulted in
     * the given category.
     *
     * @param classification The classification to learn.
     */
    public void learn(Classification<T, K> classification) {

        classification
                .getFeatureset()
                .forEach(feature -> incrementFeature(feature, classification.getCategory()));
        incrementCategory(classification.getCategory());

        memoryQueue.offer(classification);
        if (memoryQueue.size() > memoryCapacity) {
            Classification<T, K> toForget = memoryQueue.remove();

            for (T feature : toForget.getFeatureset()) {
                decrementFeature(feature, toForget.getCategory());
            }
            decrementCategory(toForget.getCategory());
        }
    }

    /**
     * The classify method. It will retrieve the most likely category for the
     * features given and depends on the concrete classifier implementation.
     *
     * @param features The features to classify.
     * @return The category most likely.
     */
    public abstract Classification<T, K> classify(Collection<T> features);

}
