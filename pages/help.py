# descrição das colunas em markdown
import streamlit as st
from utils import LOGO

st.markdown(
    f"""
    <div style="background-color:#464e5f;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">{LOGO}</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    ### Descrição das colunas
    
    |      Header     |                                             Definition                                                                                         |
    |-----------------|----------------------------------------------------------------------------------------------------|
    | URL             | O URL do personagem de quadrinhos na Marvel Wikia                                                  |
    | Name/Alias      | O nome completo ou apelido do personagem                                                            |
    | Appearances     | O número de quadrinhos em que o personagem apareceu até 30 de abril                                 |
    | Current?        | O membro está atualmente ativo em uma equipe afiliada aos Vingadores?                               |
    | Gender          | O gênero registrado do personagem                                                                  |
    | Probationary    | Às vezes, o personagem recebeu status probatório como Vingador, esta é a data em que isso aconteceu |
    | Full/Reserve    | O mês e o ano em que o personagem foi introduzido como membro efetivo ou reserva dos Vingadores    |
    | Year            | O ano em que o personagem foi introduzido como membro efetivo ou reserva dos Vingadores             |
    | Years since joining | 2015 menos o ano                                                                            |
    | Honorary        | O status do Vingador, se eles receberam status de Vingador Honorário, se estão simplesmente na Academia, ou Efetivo, caso contrário |
    | Death1          | Sim se o Vingador morreu, Não se não                                                              |
    | Return1         | Sim se o Vingador retornou de sua primeira morte, Não se não, em branco se não aplicável            |
    | Death2          | Sim se o Vingador morreu uma segunda vez após sua ressurreição, Não se não, em branco se não aplicável |
    | Return2         | Sim se o Vingador retornou de sua segunda morte, Não se não, em branco se não aplicável            |
    | Death3          | Sim se o Vingador morreu uma terceira vez após sua segunda ressurreição, Não se não, em branco se não aplicável |
    | Return3         | Sim se o Vingador retornou de sua terceira morte, Não se não, em branco se não aplicável           |
    | Death4          | Sim se o Vingador morreu uma quarta vez após sua terceira ressurreição, Não se não, em branco se não aplicável |
    | Return4         | Sim se o Vingador retornou de sua quarta morte, Não se não, em branco se não aplicável            |
    | Death5          | Sim se o Vingador morreu uma quinta vez após sua quarta ressurreição, Não se não, em branco se não aplicável |
    | Return5         | Sim se o Vingador retornou de sua quinta morte, Não se não, em branco se não aplicável            |
    | Notes           | Descrições de mortes e ressurreições.       
"""
)


