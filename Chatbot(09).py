import streamlit as st
import openai
# Initialize your OpenAI API key
openai.api_key = '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'

# Example Ayurvedic text. Replace this with your actual text.
ayurvedic_text = """
Ayurveda, the science of life, teaches that health is maintained by the balance of the three doshas: Vata, Pitta, and Kapha. Each individual has a unique balance of these forces that defines their constitution and influences their physical, mental, and emotional health. 

Diet and lifestyle routines are among the most important factors in Ayurveda, emphasizing preventive and healing therapies along with various methods of purification and rejuvenation. Ayurveda is a holistic approach that helps individuals to maintain and balance their health.

Herbs are extensively used in Ayurveda for their healing properties. For example, Turmeric (Curcuma longa) is widely recognized for its anti-inflammatory, antioxidant, and anticarcinogenic properties. Ashwagandha (Withania somnifera) is another herb that is used for its adaptogenic and stress-reducing effects.

Panchakarma is a set of five therapeutic treatments intended to purify the body. It is used to eliminate toxins from the body, restore constitutional balance, and improve health and wellness.
"""

def get_answer(question, context):
    response = openai.Completion.create(
      engine="gpt-3.5-turbo",
      prompt=f"{context}\n\nQ: {question}\nA:",
      temperature=0.5,
      max_tokens=150,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    return response.choices[0].text.strip()

# Streamlit app layout
st.title('Ayurvedic Texts QA System')

question = st.text_input('Ask a question about Ayurveda:')

if question:
    answer = get_answer(question, ayurvedic_text)
    st.text_area("Answer:", value=answer, height=300, help="Answer from the Ayurvedic texts.")