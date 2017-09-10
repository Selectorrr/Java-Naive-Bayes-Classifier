package de.daslaboratorium.machinelearning.classifier.bayes;

import de.daslaboratorium.machinelearning.classifier.Classification;
import de.daslaboratorium.machinelearning.classifier.Classifier;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.hamcrest.CoreMatchers.is;

public class BayesClassifierTest {

    private static final double EPSILON = 0.001;
    private static final String CATEGORY_NEGATIVE = "negative";
    private static final String CATEGORY_POSITIVE = "positive";
    private Classifier<String, String> bayes;

    @Before
    public void setUp() {
        /*
         * Create a new classifier instance. The context features are Strings
         * and the context will be classified with a String according to the
         * featureset of the context.
         */
        bayes = new BayesClassifier<>();

        /*
         * The classifier can learn from classifications that are handed over to
         * the learn methods. Imagin a tokenized text as follows. The tokens are
         * the text's features. The category of the text will either be positive
         * or negative.
         */
        final String[] positiveText = "I love sunny days".split("\\s");
        bayes.learn(CATEGORY_POSITIVE, Arrays.asList(positiveText));

        final String[] negativeText = "I hate rain".split("\\s");
        bayes.learn(CATEGORY_NEGATIVE, Arrays.asList(negativeText));
    }

    @Test
    public void testStringClassification() {
        final String[] unknownText1 = "today is a sunny day".split("\\s");
        final String[] unknownText2 = "there will be rain".split("\\s");

        Assert.assertThat(bayes.classify(Arrays.asList(unknownText1)).getCategory(), is(CATEGORY_POSITIVE));
        Assert.assertThat(bayes.classify(Arrays.asList(unknownText2)).getCategory(), is(CATEGORY_NEGATIVE));
    }

    @Test
    public void testStringClassificationInDetails() {

        final String[] unknownText1 = "today is a sunny day".split("\\s");

        Collection<Classification<String, String>> classifications = ((BayesClassifier<String, String>) bayes)
                .classifyDetailed(Arrays.asList(unknownText1));

        List<Classification<String, String>> list = new ArrayList<>(classifications);

        Assert.assertThat(list.get(0).getCategory(), is(CATEGORY_NEGATIVE));
        Assert.assertEquals(0.0078125, list.get(0).getProbability(), EPSILON);

        Assert.assertThat(list.get(1).getCategory(), is(CATEGORY_POSITIVE));
        Assert.assertEquals(0.0234375, list.get(1).getProbability(), EPSILON);
    }

    @Test
    public void testSerialization() throws IOException {

        new ObjectOutputStream(new ByteArrayOutputStream()).writeObject(bayes);
    }
}