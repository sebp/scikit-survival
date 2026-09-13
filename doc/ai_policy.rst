.. _ai_guidelines:

Guidelines for using AI tools
=============================

This page documents required rules when using AI tools to contribute to scikit-survival.

AI can be genuinely useful: for learning about the codebase, drafting
descriptions, thinking through a problem, or keeping track of a complex review.
The goal of this policy is not to restrict AI use, but to ensure that
contributions reflect real understanding and that AI use is transparent.

Each contribution to scikit-survival is required to undergo code review, which
requires a significant time commitment by the reviewer.
It is important that AI tools are used in a way that adds value to the project
and respects the time of other contributors and maintainers.

You can use AI extensively and produce excellent contributions. You can also
produce low-quality contributions without using AI at all. What matters is
understanding, care, and transparency, regardless of the tools involved.

Two principles must underpin every contribution:

* **Understanding**: You are responsible for what you submit. Using AI to
  generate code or text you don't understand is not a shortcut; it shifts work
  onto reviewers and maintainers without adding value. Using AI to *build*
  understanding is encouraged.
* **Transparency**: Disclose AI use clearly, so reviewers can calibrate their
  expectations and everyone stays on the same page.

.. attention::

    Maintainers may close issues and PRs that are not useful or productive, regardless of whether AI tools were used or not.


Contributor requirements
------------------------

The human driving any contribution — whether as an author, reviewer, or maintainer — must remain in control of the process
and bear full responsibility for the final output. This means actively reviewing, understanding, and validating any
AI-generated content before submission. You cannot delegate your responsibility to an AI tool, nor can you claim that
an AI made an error as a defense for problematic contributions. The human contributor is accountable for ensuring that
all content, regardless of its origin, meets project standards, follows established policies, and contributes meaningfully
to scikit-survival.

.. attention::

    Do not submit an AI-generated PR you haven't personally understood and tested, as this wastes maintainers' time.
    PRs that appear to violate this guideline will be closed without review.

* **Disclose AI use explicitly.** If AI tools contributed meaningfully to your
  PR (code, tests, or description), say so in the PR description.
  Use `GitHub's co-author feature`_
  where applicable (e.g., ``Co-authored-by: Claude <claude@anthropic.com>``).
  Please consider sharing your AI conversation log directly in the PR
  description, or link to the log in a GitHub Gist. There is no penalty for
  disclosure; there is a problem with concealment.

* **Understand the code you are submitting.** Before opening a :ref:`non-draft PR <submit-pull-request>`,
  you must be able to explain what the proposed changes do and why they are
  correct. If a reviewer asks you about your implementation and your honest
  answer is "I'm not sure, the AI did it," the PR is not ready, and your PR will
  likely be closed.

* **Review all AI-generated output before submitting.** You are responsible for
  all code or other material you submit, including its quality and
  appropriateness for contribution.

* **Verify the accuracy of AI-generated material.** This includes everything you
  contribute: PR descriptions, inline comments, and responses to reviewer
  feedback. If an AI tells you that review feedback has been addressed, verify
  that it actually has been. Do not relay AI output as your own assessment.


Acceptable uses
---------------

AI tools are well-suited to:

* Familiarizing yourself with the codebase
* Assisting with writing in a non-native language
* Talking through a design decision or debugging approach
* Assisting reviewers in understanding a large or complex diff
* Code completion, i.e. suggestions in text editors
* Identifying potential edge cases and bugs
* Explaining mathematical concepts in survival analysis to build your understanding

These are uses that support human understanding and judgment, not a substitute
for it.


Unacceptable uses
-----------------

The use of an AI agent that writes code and then submits a pull request autonomously is not permitted.
Examples include:

* Prompt an AI with a link to a GitHub issue: "Fix this and open a PR."
* Prompt an AI to generate a PR description on your behalf based on your code changes.
* Prompt an AI with a link to the PR: "Review this."
* Prompt an AI with a link to a code review: "Address this feedback."
* Prompting AI with the entire codebase to "find all bugs and fix them"
* Submitting an AI-generated implementation of an algorithm you don't understand
* Having AI interact with other contributors on your behalf (e.g., in discussions, comments)

These are uses where outputs from AI tools are the sole basis for a contribution
and no human judgment was involved.

See also
--------

* `Open edX's AI Contribution Policy <https://github.com/openedx/.github/blob/master/AI_POLICY.md>`_
* `Python's Guidelines for using AI tools <https://devguide.python.org/getting-started/ai-tools/index.html>`_
* `NumPy's AI Policy <https://numpy.org/doc/stable/dev/ai_policy.html>`_
* `Argo project Generative AI policy <https://github.com/argoproj/argoproj/blob/main/community/genai.md>`_
* `Zulips's AI use policy and guidelines <https://zulip.readthedocs.io/en/latest/contributing/contributing.html#ai-use-policy-and-guidelines>`_

.. _GitHub's co-author feature: https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/creating-a-commit-with-multiple-authors
