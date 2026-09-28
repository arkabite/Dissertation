/**
 * Plain-language explainer for first-time visitors. Keep model-specific
 * metrics out of this copy so it remains accurate after retraining.
 */
export default function AboutPanel() {
  return (
    <details className="about-panel" open>
      <summary>What is this tool?</summary>

      <p>
        This is a research prototype that lets you ask questions, in plain English, about
        an AI model that reads brain-activity recordings.
      </p>

      <h5>The short version</h5>
      <p>
        People did a memory task called the <strong>n-back task</strong> (spotting when an
        item repeats from a few steps earlier; <em>0-back</em> is the easy version,{" "}
        <em>2/3-back</em> is harder) while sensors on the scalp measured changes in blood
        oxygen in the brain (a technique called <strong>fNIRS</strong>). A rule-based AI
        then learned patterns such as{" "}
        <em>
          "IF blood oxygen at channel AF7 is High THEN the task was the easy 0-back
          version"
        </em>
        . This tool explains those rules and shows where on the head they apply.
      </p>

      <h5>What you can ask</h5>
      <ul>
        <li>
          <strong>About the model:</strong> "How reliable is this model?", "What does AF7
          tell us?"
        </li>
        <li>
          <strong>Where things are:</strong> "Where is AF7 located?", "Which channels are
          near AFp8?"
        </li>
        <li>
          <strong>How channels combine:</strong> "How are AF7 and C6h related?"
        </li>
        <li>
          <strong>General brain science:</strong> "What is the hemodynamic response
          function?"
        </li>
      </ul>

      <h5>How to read the answers</h5>
      <ul>
        <li>
          Answers about the model come from the rules listed below. Open{" "}
          <em>"Rules used"</em> under an answer to see exactly which rules were consulted.
        </li>
        <li>
          A blue <em>"General knowledge"</em> badge means textbook-style context,{
          " "
          }<strong>not</strong> something this model found. Sources are shown when a
          matching paper is available.
        </li>
        <li>
          On the maps, <strong>colour</strong> is the signal level (Low / Medium / High),
          a <strong>circle</strong> is oxygenated blood (HbO) and a{" "}
          <strong>diamond</strong> is deoxygenated blood (HbR).
        </li>
      </ul>

      <h5>Jargon, briefly</h5>
      <ul>
        <li>
          <strong>Channel</strong> - one measurement spot on the scalp (e.g. AF7).
        </li>
        <li>
          <strong>DS</strong> (dominance score) - roughly, how much weight a rule carries
          in the model's decisions. <strong>ACC</strong> - how often that rule was right
          on the trials it applied to.
        </li>
        <li>
          <strong>MCC</strong> - an overall score for the whole model where 0 is no better
          than guessing and 1 is perfect.
        </li>
      </ul>

      <h5>Please keep in mind</h5>
      <ul>
        <li>This is a research prototype, not a medical or diagnostic tool.</li>
        <li>
          Answers are written by an AI language model and can contain mistakes. When it
          matters, check them against the rules shown.
        </li>
        <li>
          The rules come from one trained model and can change if it is retrained; see{" "}
          <em>Performance</em> below for how well it does.
        </li>
        <li>It only knows about this one model and dataset.</li>
      </ul>
    </details>
  );
}