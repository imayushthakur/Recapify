class PromptTemplates:
    @staticmethod
    def video_recap_template():
        """Return a template for video recapitulation."""
        return """
I want you to act as a tutorial summarizer that can create concise and informative summaries of educational videos.

Here are some examples of tutorial transcripts and their summaries:

{examples}

Now, create a summary for the following tutorial transcript. Focus on:
1. Main topics covered
2. Key points for each topic
3. Important techniques or methods explained
4. Any practical tips or best practices mentioned

Make sure the summary is well-structured, informative, and captures the educational value of the original tutorial without watching it.

Tutorial Transcript:
{input}
"""
